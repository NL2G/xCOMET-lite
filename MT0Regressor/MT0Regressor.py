import torch
import torch.nn as nn
from transformers import MT5EncoderModel
from transformers.modeling_outputs import BaseModelOutput

class Args():
    def __init__(self, encoder_name, sizes_mlp, hidden_act, dropout_coef, 
                 size_labsell, need_lora, output_act, loss_fc, use_labse, checkpoint: str = None):
        self.encoder_name = encoder_name
        self.sizes_mlp = sizes_mlp
        self.size_labsell = size_labsell
        self.hidden_act = hidden_act
        self.dropout_coef = dropout_coef
        self.need_lora = need_lora
        self.output_act = output_act
        self.loss_fc = loss_fc
        self.use_labse = use_labse
        self.checkpoint = checkpoint
        print(f"USE_LABSE == {use_labse}")


def mean_pooling(token_embeddings, attention_mask):    
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


class MT0Regressor(nn.Module):
    def __init__(self, config: Args):
        super(MT0Regressor, self).__init__()

        self.llm = MT5EncoderModel.from_pretrained(
            config.encoder_name, output_attentions=True, output_hidden_states=True)
        
        if config.checkpoint is not None:
            checkpoint = torch.load(config.checkpoint)
            checkpoint = {key.replace("module.", ""):value for key, value in checkpoint.items()}
            self.llm.load_state_dict(checkpoint)
            print(f"Checkpoint loaded")

        dropout_coef = config.dropout_coef

        self.use_labse = config.use_labse

        if self.use_labse:
            self.mlp_labse = nn.Linear(768*3, config.size_labsell)
            layers = [nn.Linear(1024 + config.size_labsell, config.sizes_mlp[0]), config.hidden_act()]
        else:
            layers = [nn.Linear(1024, config.sizes_mlp[0]), config.hidden_act()] # last hidden layer size + labse vector size
        for i in range(len(config.sizes_mlp) - 1):
            layers.append(nn.Linear(config.sizes_mlp[i],
                                    config.sizes_mlp[i + 1]))
            if i < len(config.sizes_mlp) - 2:
                layers.append(config.hidden_act())

        self.dropout_input = nn.Dropout(dropout_coef)
        self.mlp = nn.Sequential(*layers)
        self.output_act = config.output_act()

        self.loss_fc = config.loss_fc()

    def forward(self, input_ids=None, attention_mask=None, labels=None, labse=None):
        outputs = self.llm(input_ids=input_ids, attention_mask=attention_mask)
        embeddings_mt0 = mean_pooling(
            outputs.last_hidden_state, attention_mask
        )

        if self.use_labse:
            vector_tip = self.mlp_labse(labse)
            full_emb = torch.cat((embeddings_mt0, vector_tip), -1)
        else:
            full_emb = embeddings_mt0

        outputs_sequence = self.dropout_input(full_emb)

        logits = self.output_act(self.mlp(outputs_sequence))

        loss = None
        if labels is not None:
            loss = self.loss_fc(logits.view(-1, 1),
                                labels.view(-1).unsqueeze(1))

        return (
            BaseModelOutput(
                last_hidden_state=outputs.last_hidden_state,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            ),
            logits,
            loss,
        )
