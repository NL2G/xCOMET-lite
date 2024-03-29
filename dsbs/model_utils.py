# Copyright 2023 The Distilling-step-by-step authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
from torch import nn
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


def length_fn(examples):
    return {'length': [len(x) for x in examples['input_ids']]}


def get_tokenize_fn(tokenizer: PreTrainedTokenizer, kind: str = '1way'):
    
    def tokenize_function_1way(examples):
        model_inputs = tokenizer(examples['input'], max_length=1024, truncation=True)
        
        labels = tokenizer(text_target=examples['class'], max_length=16, truncation=True)
    
        model_inputs['labels'] = labels['input_ids']
    
        return model_inputs

    def tokenize_function_2way(examples):
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=1024, truncation=True)
        expl_model_inputs = tokenizer(['explain: ' + text for text in examples['input']], max_length=1024, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
    
        label_output_encodings = tokenizer(text_target=examples['class'], max_length=16, truncation=True)
        rationale_output_encodings = tokenizer(text_target=examples['rationale'], max_length=256, truncation=True)

        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['expl_labels'] = rationale_output_encodings['input_ids']
    
        return model_inputs

    def tokenize_function_3way(examples):
        model_inputs = tokenizer(['predict: ' + text for text in examples['input']], max_length=1024, truncation=True)
        expl_model_inputs = tokenizer(['explain good: ' + text for text in examples['input']], max_length=1024, truncation=True)
        antiexpl_model_inputs = tokenizer(['explain bad: ' + text for text in examples['input']], max_length=1024, truncation=True)
        model_inputs['expl_input_ids'] = expl_model_inputs['input_ids']
        model_inputs['expl_attention_mask'] = expl_model_inputs['attention_mask']
        model_inputs['antiexpl_input_ids'] = antiexpl_model_inputs['input_ids']
        model_inputs['antiexpl_attention_mask'] = antiexpl_model_inputs['attention_mask']
    
        label_output_encodings = tokenizer(text_target=examples['class'], max_length=16, truncation=True)
        rationale_output_encodings = tokenizer(text_target=examples['rationale'], max_length=256, truncation=True)
        antirationale_output_encodings = tokenizer(text_target=examples['antirationale'], max_length=256, truncation=True)
    
        model_inputs['labels'] = label_output_encodings['input_ids']
        model_inputs['expl_labels'] = rationale_output_encodings['input_ids']
        model_inputs['antiexpl_labels'] = antirationale_output_encodings['input_ids']
    
        return model_inputs

    fns = {
        '1way': tokenize_function_1way,
        '2way': tokenize_function_2way,
        '3way': tokenize_function_3way
    }
    assert kind in fns.keys(), f"Kind can only be one of: {fns.keys()}"
    return fns[kind]
        

class DataCollator1Way(DataCollatorForSeq2Seq):

    def __call__(self, inputs):
        pred_features = super().__call__([
            {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels']
            } for item in inputs
        ])
        return {'pred': pred_features}

class DataCollator2Way(DataCollatorForSeq2Seq):

    def __call__(self, inputs):
        pred_features = super().__call__([
            {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels']
            } for item in inputs
        ])
        expl_features = super().__call__([
            {
                'input_ids': item['expl_input_ids'],
                'attention_mask': item['expl_attention_mask'],
                'labels': item['expl_labels']
            } for item in inputs
        ])
        return {
            'pred': pred_features,
            'expl': expl_features
        }

class DataCollator3Way(DataCollatorForSeq2Seq):

    def __call__(self, inputs):
        pred_features = super().__call__([
            {
                'input_ids': item['input_ids'],
                'attention_mask': item['attention_mask'],
                'labels': item['labels']
            } for item in inputs
        ])
        expl_features = super().__call__([
            {
                'input_ids': item['expl_input_ids'],
                'attention_mask': item['expl_attention_mask'],
                'labels': item['expl_labels']
            } for item in inputs
        ])
        antiexpl_features = super().__call__([
            {
                'input_ids': item['antiexpl_input_ids'],
                'attention_mask': item['antiexpl_attention_mask'],
                'labels': item['antiexpl_labels']
            } for item in inputs
        ])
        return {
            'pred': pred_features,
            'expl': expl_features,
            'antiexpl': antiexpl_features
        }
    

KIND_TO_DATACOLLATOR = {
    '1way': DataCollator1Way,
    '2way': DataCollator2Way,
    '3way': DataCollator3Way
}


class TaskPrefixTrainer(Seq2SeqTrainer):
    def __init__(
        self, alpha1: float = 0.0,
        alpha2: float = 0.0, 
        kind: str = '1way', 
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        assert kind in {'1way', '2way', '3way'}
        self.kind = kind


    def compute_loss(self, model, inputs, return_outputs=False):
        if self.kind == '1way':
            pred_outputs = model(**inputs['pred'])
            loss = pred_outputs.loss
            return (
                loss,
                {
                    'pred': pred_outputs
                }
            ) if return_outputs else loss
        elif self.kind == '2way':
            pred_outputs = model(**inputs['pred'])
            expl_outputs = model(**inputs['expl'])
            loss = self.alpha1 * pred_outputs.loss + (1. - self.alpha1) * expl_outputs.loss
            return (
                loss,
                {
                    'pred': pred_outputs,
                    'expl': expl_outputs
                }
            ) if return_outputs else loss
        else:
            # 3way
            pred_outputs = model(**inputs['pred'])
            expl_outputs = model(**inputs['expl'])
            antiexpl_outputs = model(**inputs['antiexpl'])
            loss = self.alpha1 * pred_outputs.loss + self.alpha2 * expl_outputs.loss + (1 - self.alpha1 - self.alpha2) * antiexpl_outputs.loss
            return (
                loss,
                {
                    'pred': pred_outputs,
                    'expl': expl_outputs,
                    'antiexpl': antiexpl_outputs
                }
            ) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:

        pred_outputs = super().prediction_step(model, inputs['pred'], prediction_loss_only=False, ignore_keys=ignore_keys)
        loss = pred_outputs[0]
        logger.info(f"Loss: {loss}")
        return (
            loss,
            pred_outputs[1],
            pred_outputs[2]
        )
