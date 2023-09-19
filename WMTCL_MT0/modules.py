import os
import numpy as np
import gc
import faiss
from tqdm import tqdm

import torch
from torch import nn
import datasets
from sentence_transformers import SentenceTransformer

from datasets import Dataset

from specials import *


def free_():
    gc.collect()
    torch.cuda.empty_cache()


def add_info_to_path(parts, info):
    return f'{parts[0]}_{info}.ckpt' if len(parts) == 1 else f'{parts[0]}_{info}.{parts[1]}'


# def save_model(model, accelerator, filepath, final=False):
#     if accelerator.is_local_main_process:
#         if final:
#             accelerator.save(model.state_dict(), filepath)
#             return
#         i = 0
#         parts = filepath.rsplit('.', 1)
#         for _ in range(N_COPIES):
#             path_ = add_info_to_path(parts, i)
#             if not os.path.exists(path_):
#                 break
#             i += 1
#         if i < N_COPIES:
#             accelerator.save(model.state_dict(), add_index_to_path(parts, i))
#             return
#         os.remove(add_index_to_path(parts, 0))
#         for j in range(1, N_COPIES):
#             os.rename(add_index_to_path(parts, j), add_index_to_path(parts, j-1))

#         accelerator.save(model.state_dict(), add_index_to_path(parts, N_COPIES-1))

def save_model(model, accelerator, filepath, n_steps, eval_val, final=False):
    if accelerator.is_local_main_process:
        if final:
            accelerator.save(model.state_dict(), filepath)
            return
        parts = filepath.rsplit('.', 1)
        info = f'{n_steps}_{eval_val}'
        filepath_ = add_info_to_path(parts, info)
        if os.path.exists(filepath_):
            os.remove(filepath_)
        accelerator.save(model.state_dict(), filepath_)


def prepare_faiss(
    model,
    pool,
    sentences,
    nlist=100, # voronoi cells for ANN
    m=16, # number of centroid IDs in final compressed vectors
    bits=8,  # number of bits in each centroid
    nprobe=10, # number of cells to search during inference,
    save_filepath=None
):
    embs = model.encode_multi_process(sentences, pool, batch_size=128)
    embs = -embs / np.linalg.norm(embs, axis=1, keepdims=True)

    d = embs.shape[1]
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits)
    index.train(embs)
    index.add(embs)
    index.nprobe = nprobe
    if save_filepath is not None:
        faiss.write_index(index, save_filepath)
    return index


def get_lp_training_staff(train_dset, st_model):
    lp2train_dset = {}
    lp2ref_index = {}
    lp2mt_index = {}

    st, pool = None, None
    for lp in LANG_PAIRS:

        dset_path = f'{DATA_DIR}/{lp}_dset.pt'
        if os.path.exists(dset_path):
            lp_dset = torch.load(dset_path)
        else:
            st = SentenceTransformer(st_model)
            pool = st.start_multi_process_pool()
            lp_dset = train_dset.filter(lambda x: x['lp'] == lp).shuffle(seed=SEED)
            src_embs_dset = Dataset.from_dict({'src_emb': st.encode_multi_process(lp_dset['src'], pool, batch_size=128)})
            lp_dset = datasets.concatenate_datasets([lp_dset, src_embs_dset], axis=1)
            torch.save(lp_dset, dset_path)
        lp2train_dset[lp] = lp_dset

        ref_index_path = f'{DATA_DIR}/{lp}_ref_faiss.idx'
        mt_index_path = f'{DATA_DIR}/{lp}_mt_faiss.idx'

        if os.path.exists(ref_index_path):
            lp2ref_index[lp] = faiss.read_index(ref_index_path)
        else:
            if st is None:
                st = SentenceTransformer(st_model)
                pool = st.start_multi_process_pool()
            lp2ref_index[lp] = prepare_faiss(st, pool, lp_dset['ref'], save_filepath=ref_index_path)

        if os.path.exists(mt_index_path):
            lp2mt_index[lp] = faiss.read_index(mt_index_path)
        else:
            if st is None:
                st = SentenceTransformer(st_model)
                pool = st.start_multi_process_pool()
            lp2mt_index[lp] = prepare_faiss(st, pool, lp_dset['mt'], save_filepath=mt_index_path)

    if st is not None:
        st.stop_multi_process_pool(pool)
    return lp2train_dset, lp2ref_index, lp2mt_index


def preprocess(text, lang=None):
    if text is None:
        return ''
    text = text.lower().strip()
    if lang is not None:
        text = f'{lang}: {text}'
    return ' '.join(text.split())


def tokenize_(data, tokenizer, max_length=MAX_LENGTH, add_lang=False):
    src_lang, tgt_lang = map(lambda x: ID2LANG[x], data['lp'].split('-'))
    output = {}
    for field in ['src', 'ref', 'mt']:
        result = tokenizer(preprocess(data[field], src_lang if field == 'src' else tgt_lang),
                           truncation=True, max_length=max_length, padding=False)
        if max_length is not None and result['input_ids'][-1] != tokenizer.eos_token_id \
            and len(result['input_ids']) < max_length:
            result['input_ids'].append(tokenizer.eos_token_id)
            result['attention_mask'].append(1)
        output[f'{field}_input_ids'] = result['input_ids']
        output[f'{field}_attention_mask'] = result['attention_mask']
    return output


class DatasetWMTCL(Dataset):

    def __init__(
        self,
        input,
        inference=False,
        train_batch_size=32
    ):
        self.inference = inference
        assert isinstance(input, list) and (not inference or len(input) == 1)
        if inference:
            self.datasets = input
            self.total_len = len(input[0])
            return

        self.total_len = 0
        self.cumsum_lens = []
        self.datasets = []
        self.ref_indexes = []
        self.mt_indexes = []
        self.train_batch_size = train_batch_size
        self.n_neighbors = (train_batch_size - 3) // 2

        for part in input:
            dataset, ref_index, mt_index = part
            self.ref_indexes.append(ref_index)
            self.mt_indexes.append(mt_index)
            self.datasets.append(dataset)
            self.total_len += len(dataset)
            self.cumsum_lens.append(self.total_len)

    def __len__(self):
        return self.total_len

    def determine_data_index(self, idx):
        prev_cumsum_len = 0
        for i, cumsum_len in enumerate(self.cumsum_lens):
            if idx < cumsum_len:
                return i, idx - prev_cumsum_len
            prev_cumsum_len = cumsum_len
        raise ValueError(f'Index {idx} is not in valid range')

    def __getitem_for_train(self, idx):
        assert len(idx) == 1
        idx = idx[0]

        output = {}
        i, idx = self.determine_data_index(idx)
        dataset = self.datasets[i]
        point = dataset[idx]
        output = {'score_type': [point['score_type']]*self.train_batch_size, 'score': [point['score']]*self.train_batch_size}
        for key in ['input_ids', 'attention_mask']:
            output[key] = [point[f'{field}_{key}'] for field in ['src', 'ref', 'mt']]

        src_emb = np.asarray(point['src_emb'])[None, :]
        ref_index = self.ref_indexes[i]
        mt_index = self.mt_indexes[i]

        _, ref_I = ref_index.search(src_emb, k=self.n_neighbors+1+1)
        _, mt_I = mt_index.search(src_emb, k=self.n_neighbors+1)

        ref_I = ref_I.ravel()
        mt_I = mt_I.ravel()

        added_ref = 0
        added_mt = 0
        for j in range(self.n_neighbors+1+1):
            if ref_I[j] == idx:
                continue
            far_point = dataset[int(ref_I[j])]
            for key in ['input_ids', 'attention_mask']:
                output[key].append(far_point[f'ref_{key}'])
            added_ref += 1
            if added_ref == self.n_neighbors + 1:
                break
        for j in range(self.n_neighbors+1):
            if mt_I[j] == idx:
                continue
            far_point = dataset[int(mt_I[j])]
            for key in ['input_ids', 'attention_mask']:
                output[key].append(far_point[f'mt_{key}'])
            added_mt += 1
            if added_mt == self.n_neighbors:
                break

        return output

    def __getiten_for_test(self, idx):
        output = {}
        dataset = self.datasets[0]
        points = dataset[idx]
        for key in ['input_ids', 'attention_mask']:
            output[key] = []
            for field in ['src', 'ref']:
                output[key] += points[f'{field}_{key}']
        return output

    def __getitem__(self, idx):
        return self.__getitem_for_train(idx) if not self.inference else self.__getiten_for_test(idx)


class DataCollatorWithPaddingAndScore:

    def __init__(
        self,
        tokenizer,
        padding=True,
        max_length=None,
        pad_to_multiple_of=None,
        return_tensors="pt"
    ):
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors

    def __call__(self, features):
        scores = []
        score_types = []
        for feature in features:
            scores.append(feature.pop('score'))
            score_types.append(feature.pop('score_type'))
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch['score'] = scores
        batch['score_type'] = score_types
        return batch


class ContrastiveLossWMT(nn.Module):
    """
    Full credit to https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/.
    """
    def __init__(
        self,
        negative_n, # n_neighbors from dataset initialization
        score_type_weights=None, # prioritization of score types
        temperature=0.05,
        device='cpu'
    ):
        super().__init__()
        self.negative_n = negative_n
        self.score_type_weights = score_type_weights
        self.register_buffer("temperature", torch.tensor(temperature, device=device))

    def forward(self, embs: torch.Tensor, score: float, score_type: str):
        embs = torch.nn.functional.normalize(embs, dim=1)
        src_emb = embs[0:1]
        tgt_embs = embs[1:]
        similarity_vector = src_emb @ tgt_embs.T
        similarity_vector = similarity_vector.squeeze()

        ref_nom  = torch.exp(similarity_vector[0] / self.temperature)
        ref_denom = ref_nom + torch.exp(similarity_vector[2 : 2+self.negative_n+1] / self.temperature).sum()
        mt_nom  = self.score_type_weights[score_type] * score * torch.exp(similarity_vector[1] / self.temperature)
        mt_denom = mt_nom + torch.exp(similarity_vector[2+self.negative_n+1 : 2+2*self.negative_n+1] / self.temperature).sum()
        loss = -torch.log(ref_nom / ref_denom) - torch.log(mt_nom / mt_denom)
        return loss
