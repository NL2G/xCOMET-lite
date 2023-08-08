import numpy as np
import torch
import torch.nn as nn

from .encoder import Encoder
from .modules import FeedForward, LayerwiseAttention


def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Method that masks embeddings representing padded elements.

    Args:
        fill_value (float): the value to fill the embeddings belonging to padded tokens
        tokens (torch.Tensor): Word ids [batch_size x seq_length]
        embeddings (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        padding_index (int):Padding value.

    Return:
        torch.Tensor: Word embeddings [batch_size x seq_length x hidden_size]
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)


def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Average pooling method.

    Args:
        tokens (torch.Tensor): Word ids [batch_size x seq_length]
        embeddings (torch.Tensor): Word embeddings [batch_size x seq_length x
            hidden_size]
        mask (torch.Tensor): Padding mask [batch_size x seq_length]
        padding_index (torch.Tensor): Padding value.

    Return:
        torch.Tensor: Sentence embedding
    """
    wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask


class Comet(nn.Module):
    def __init__(
        self,
        encoder_model_name: str,
        use_adapters: bool = False,
        adapter_config: str = "pfeiffer",
        layer: int | str = "mix",
        keep_embeddings_freezed: bool = True,
        hidden_sizes: list[int] = [3072, 1024],
        activations: str = "Tanh",
        final_activation: str | None = None,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            encoder_model_name, use_adapters=use_adapters, adapter_config=adapter_config
        )
        self.estimator = FeedForward(
            in_dim=self.encoder.output_dim * 6,
            hidden_sizes=hidden_sizes,
            activations=activations,
            final_activation=final_activation,
            out_dim=1,
            dropout=dropout,
        )

        self.layer = layer
        if self.layer == "mix":
            self.layerwise_attention = LayerwiseAttention(
                num_layers=self.encoder.num_layers
            )
        else:
            self.layerwise_attention = None

        if keep_embeddings_freezed:
            self.encoder.freeze_embeddings()

        self.pad_token_id = pad_token_id

    def compute_embeddings(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        encoder_out = self.encoder(input_ids, attention_mask)
        if self.layerwise_attention:
            embeddings = self.layerwise_attention(
                encoder_out["all_layers"], attention_mask
            )
        else:
            embeddings = encoder_out["all_layers"][self.layer]

        embeddings = average_pooling(
            input_ids, embeddings, attention_mask, self.pad_token_id
        )

        return embeddings

    def estimate(
        self,
        src_sentemb: torch.Tensor,
        mt_sentemb: torch.Tensor,
        ref_sentemb: torch.Tensor,
    ) -> torch.Tensor:
        """Method that takes the sentence embeddings from the Encoder and runs the
        Estimator Feed-Forward on top.

        Args:
            src_sentemb [torch.Tensor]: Source sentence embedding
            mt_sentemb [torch.Tensor]: Translation sentence embedding
            ref_sentemb [torch.Tensor]: Reference sentence embedding

        Return:
            Prediction object with sentence scores.
        """
        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
            dim=1,
        )

        scores = self.estimator(embedded_sequences).view(-1)
        return scores

    def forward(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: torch.Tensor,
        mt_input_ids: torch.Tensor,
        mt_attention_mask: torch.Tensor,
        ref_input_ids: torch.Tensor,
        ref_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        src_sentemb = self.compute_embeddings(src_input_ids, src_attention_mask)
        mt_sentemb = self.compute_embeddings(mt_input_ids, mt_attention_mask)
        ref_sentemb = self.compute_embeddings(ref_input_ids, ref_attention_mask)

        return self.estimate(src_sentemb, mt_sentemb, ref_sentemb)
