import torch
import torch.nn as nn
import transformers as tr
import peft



class Encoder(nn.Module):
    def __init__(
        self, 
        model_name: str,
        model_class: str = "AutoModel",
        model_args: dict = None,
        n_bits: int = 8,
        adapter_type: str = 'LoRA', 
        adapter_config: str = None
    ) -> None:
        super().__init__()

        model_class_instance = getattr(tr, model_class)
        if model_args is None:
            model_args = {}

        if adapter_config is None:
            adapter_config = {}
        
        if n_bits == 32:
            model_args['torch_dtype'] = torch.float32
        elif n_bits == 16:
            model_args['torch_dtype'] = torch.bfloat16
        elif n_bits == 8:
            config = tr.BitsAndBytesConfig(load_in_8bit=True)
            model_args['quantization_config'] = config
            model_args['torch_dtype'] = torch.float16
        elif n_bits == 4:
            model_args["quantization_config"] = tr.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_args['torch_dtype'] = torch.bfloat16
        else:
            raise ValueError("Invalid number of bits")
        
        self.model = model_class_instance.from_pretrained(
            model_name,
            **model_args
        )

        if n_bits == 4:
            self.model.gradient_checkpointing_enable()
            self.model = peft.prepare_model_for_kbit_training(self.model)
        elif n_bits == 8:
            self.model = peft.prepare_model_for_int8_training(self.model)

        if adapter_type == 'LoRA':
            peft_config = peft.LoraConfig(task_type=peft.TaskType.FEATURE_EXTRACTION, **adapter_config)
        elif adapter_type == 'IA3':
            peft_config = peft.IA3Config(task_type=peft.TaskType.FEATURE_EXTRACTION, **adapter_config)
        else:
            raise ValueError("Invalid adapter type")

        self.model = peft.get_peft_model(self.model, peft_config)

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
