from transformers import Trainer

def compute_direct_mqm(error_tags):
    pass


def aggregate_mqm(src_mqm, ref_mqm, src_ref_mqm, direct_mqm):
    mqm = 2/9 * direct_mqm
    total_weights = 2/9

    if src_mqm is not None:
        mqm = mqm + 1/9 * src_mqm
        total_weights += 1/9
    
    if ref_mqm is not None:
        mqm = mqm + 1/3 * ref_mqm
        total_weights += 1/3
    
    if src_ref_mqm is not None:
        mqm = mqm + 1/3 * src_ref_mqm
        total_weights += 1/3
    
    # Renormalize
    mqm = mqm / total_weights

    return mqm
    

class MQMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        source = inputs.get("source")
        machine_translation = inputs.get("machine_translation")
        reference = inputs.get("reference")
        per_token_error_tags = inputs.get("per_token_error_tags")
        mqm_score = inputs.get("mqm_score")

        src_mqm, ref_mqm, src_ref_mqm = None, None, None

        tag_criterion = nn.CrossEntropyLoss(weights=...)

        if source is not None:
            # compute source-only
            src_mqm, src_error_tags = model(...)
            src_tag_loss = tag_criterion(per_token_error_tags, src_error_tags)

        if reference is not None:
            # compute reference-only
            ref_mqm, ref_error_tags = model(...)
            ref_tag_loss = tag_criterion(per_token_error_tags, ref_error_tags)
        
        if source is not None and reference is not None:
            # compute full
            src_ref_mqm, src_ref_error_tags = model(...)
            src_ref_tag_loss = tag_criterion(per_token_error_tags, src_ref_error_tags)

        direct_mqm = compute_direct_mqm(...)

        mqm = aggregate_mqm(src_mqm, ref_mqm, src_ref_mqm, direct_mqm)



        return (loss, {"all_logits": all_logits}) if return_outputs else loss