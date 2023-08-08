import torch
import torch.nn as nn
import transformers as tr
import transformers.adapters as atr

ADAPTER_TASK_NAME: str = "encoder-adapter"


class Encoder(nn.Module):
    def __init__(
        self, model_name: str, use_adapters: bool = False, adapter_config: str = None
    ) -> None:
        super().__init__()
        self.use_adapters = use_adapters
        if self.use_adapters:
            aconfig = atr.AdapterConfig.load(adapter_config)
            self.model = tr.AutoModel.from_pretrained(
                model_name, add_pooling_layer=False
            )
            self.model.add_adapter(ADAPTER_TASK_NAME, config=aconfig)
            self.model.train_adapter(ADAPTER_TASK_NAME)
        else:
            self.model = tr.AutoModel.from_pretrained(
                model_name, add_pooling_layer=False
            )

        self.output_dim: int = self.model.config.hidden_size
        self.num_layers: int = self.model.config.num_hidden_layers + 1
        self.max_positions: int = self.model.config.max_position_embeddings
        self.size_sepatator: int = 2  # we only use xlm-roberta models

        self.model.encoder.output_hidden_states = True

    def freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = True

        if self.use_adapters:
            self.model.train_adapter(ADAPTER_TASK_NAME)

    def freeze_embeddings(self) -> None:
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

    def layerwise_lr(self, lr: float, decay: float):
        """Calculates the learning rate for each layer by applying a small decay.

        Args:
            lr (float): Learning rate for the highest encoder layer.
            decay (float): decay percentage for the lower layers.

        Returns:
            list: List of model parameters for all layers and the corresponding lr.
        """
        # Embedding Layer
        opt_parameters = [
            {
                "params": self.model.embeddings.parameters(),
                "lr": lr * decay ** (self.num_layers),
            }
        ]
        # All layers
        opt_parameters += [
            {
                "params": self.model.encoder.layer[i].parameters(),
                "lr": lr * decay**i,
            }
            for i in range(self.num_layers - 2, 0, -1)
        ]
        return opt_parameters

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        last_hidden_states, _, all_layers = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }
