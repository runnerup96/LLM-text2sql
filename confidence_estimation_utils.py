import torch

def maximum_entropy_confidence_score_method(generation_scores, device):
    # TODO: Work with beans to samples ratio here
    logits = torch.stack(generation_scores, dim=1)[:: 1]
    logits = logits.cpu() if "cuda" in device else logits
    probs = torch.softmax(logits, dim=2).float()
    log_probs = torch.log_softmax(logits, dim=2).float()
    entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()

    return entropies

def truncate_scores(generated_sequences, scores, tokenizer):
    scores_list = []
    for idx in range(len(generated_sequences)):
        pred_tensor = generated_sequences[idx][1:]
        scores_truncated = scores[idx].tolist()

        # Truncate the prediction at the end-of-sequence token, if present.
        if tokenizer.eos_token_id in pred_tensor:
            pred_eos_idx = torch.nonzero(pred_tensor == tokenizer.eos_token_id)[0].item()
            scores_truncated = scores_truncated[: pred_eos_idx + 1]

        scores_list.append(scores_truncated)

    return scores_list