import torch

def maximum_entropy_confidence_score_method(generation_scores, device):
    """
    Calculate the maximum entropy confidence scores for generated sequences.

    Args:
        generation_scores (list of torch.Tensor): A list of tensors containing the generation scores.
        device (str): The device on which the tensors are located (e.g., 'cuda' or 'cpu').

    Returns:
        numpy.ndarray: An array of entropy values calculated from the generation scores.
    """
    logits = torch.stack(generation_scores, dim=1)[:: 1]
    logits = logits.cpu() if "cuda" in device else logits
    probs = torch.softmax(logits, dim=2).float()
    log_probs = torch.log_softmax(logits, dim=2).float()
    entropies = (torch.sum(probs * log_probs, axis=2) * (-1)).numpy()

    return entropies

def truncate_scores(generated_sequences, scores, tokenizer):
    """
    Truncate the scores at the end-of-sequence token for each generated sequence.

    Args:
        generated_sequences (list of torch.Tensor): A list of tensors representing the generated sequences.
        scores (list of list of float): A list of score lists corresponding to the generated sequences.
        tokenizer (Tokenizer): The tokenizer used to tokenize the sequences.

    Returns:
        list of list of float: A list of truncated score lists, truncated at the end-of-sequence token.
    """
    scores_list = []
    for idx in range(len(generated_sequences)):
        pred_tensor = generated_sequences[idx][1:]
        scores_truncated = scores[idx].tolist()

        if tokenizer.eos_token_id in pred_tensor:
            pred_eos_idx = torch.nonzero(pred_tensor == tokenizer.eos_token_id)[0].item()
            scores_truncated = scores_truncated[: pred_eos_idx + 1]

        scores_list.append(scores_truncated)

    return scores_list