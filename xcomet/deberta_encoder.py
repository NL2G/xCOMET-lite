from comet.encoders.base import Encoder
from comet.encoders.bert import BERTEncoder
import transformers as tr
import torch


class DeBERTaEncoder(BERTEncoder):

    def __init__(
        self, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> None:
        super(Encoder, self).__init__()
        self.tokenizer = tr.AutoTokenizer.from_pretrained(pretrained_model)
        if load_pretrained_weights:
            self.model = tr.AutoModel.from_pretrained(
                pretrained_model
            )
        else:
            self.model = tr.AutoModel(
                tr.AutoConfig.from_pretrained(pretrained_model),
            )
        self.model.encoder.output_hidden_states = True

    @property
    def size_separator(self):
        """Number of tokens used between two segments. For BERT is just 1 ([SEP])
        but models such as XLM-R use 2 (</s></s>)"""
        return 1

    @property
    def uses_token_type_ids(self):
        return True

    @classmethod
    def from_pretrained(
        cls, pretrained_model: str, load_pretrained_weights: bool = True
    ) -> Encoder:
        """Function that loads a pretrained encoder from Hugging Face.

        Args:
            pretrained_model (str):Name of the pretrain model to be loaded.
            load_pretrained_weights (bool): If set to True loads the pretrained weights
                from Hugging Face

        Returns:
            Encoder: XLMREncoder object.
        """
        return DeBERTaEncoder(pretrained_model, load_pretrained_weights)

    def forward(
        self,
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        **kwargs
    ) -> dict[str, torch.Tensor]:
        model_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get("token_type_ids", None),
            output_hidden_states=True,
        )
        return {
            "sentemb": model_output.last_hidden_state[:, 0, :],
            "wordemb": model_output.last_hidden_state,
            "all_layers": model_output.hidden_states,
            "attention_mask": attention_mask,
        }