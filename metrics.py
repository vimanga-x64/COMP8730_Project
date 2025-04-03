import re

def word_edit_distance(predicted: str, target: str) -> int:
    """
    Computes the Levenshtein edit distance between two sequences of words.
    Both predicted and target should be strings. The function splits them
    into words (using whitespace) and computes the minimum number of
    insertions, deletions, and substitutions required to transform the
    predicted word list into the target word list.
    """
    pred_words = predicted.split()
    target_words = target.split()
    n = len(pred_words)
    m = len(target_words)

    # Initialize DP table.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if pred_words[i - 1] == target_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j],  # deletion
                                   dp[i][j - 1],  # insertion
                                   dp[i - 1][j - 1])  # substitution
    return dp[n][m]


def average_word_edit_distance(predictions: list[str], targets: list[str]) -> float:
    """
    Computes the average Levenshtein edit distance between lists of predicted glosses
    and target glosses.

    Each prediction and corresponding target is assumed to be a string.
    The edit distance for each pair is computed using the word_edit_distance function,
    and the average edit distance over all pairs is returned.

    Args:
        predictions (list of str): List of predicted gloss strings.
        targets (list of str): List of target (gold) gloss strings.

    Returns:
        float: The average edit distance over the dataset.

    Raises:
        ValueError: If the lengths of the prediction and target lists do not match.
    """
    if len(predictions) != len(targets):
        raise ValueError("The number of predictions and targets must be the same.")

    total_distance = 0
    for pred, target in zip(predictions, targets):
        total_distance += word_edit_distance(pred, target)

    return total_distance / len(predictions)


def compute_word_level_gloss_accuracy(predictions: list, targets: list) -> dict:
    """
    Computes word-level glossing accuracy over a set of predictions.
    For each prediction–target pair (both as strings), it splits into tokens,
    compares tokens in order (ignoring any predicted '[UNK]'), and computes:
      - average_accuracy: average of per-example accuracies
      - accuracy: overall token accuracy across the dataset

    Args:
        predictions (list): List of predicted gloss strings.
        targets (list): List of target (gold) gloss strings.

    Returns:
        dict: A dictionary with keys 'average_accuracy' and 'accuracy'.
    """
    if not targets:
        return {'average_accuracy': 1.0, 'accuracy': 1.0}

    total_correct = 0
    total_tokens = 0
    summed_accuracies = 0.0

    for pred, target in zip(predictions, targets):
        pred_tokens = pred.split()
        target_tokens = target.split()
        entry_correct = 0
        for i in range(len(target_tokens)):
            if i < len(pred_tokens) and pred_tokens[i] == target_tokens[i] and pred_tokens[i] != '<unk>':
                entry_correct += 1
        summed_accuracies += entry_correct / len(target_tokens)
        total_correct += entry_correct
        total_tokens += len(target_tokens)

    return {'average_accuracy': summed_accuracies / len(targets),
            'accuracy': total_correct / total_tokens}


def compute_morpheme_level_gloss_accuracy(predictions: list, targets: list) -> dict:
    """
    Computes morpheme-level glossing accuracy over a set of predictions.
    For each prediction–target pair (both as strings), it splits into tokens,
    compares tokens in order (ignoring any predicted '[UNK]'), and computes:
      - average_accuracy: average of per-example accuracies
      - accuracy: overall token accuracy across the dataset

    Args:
        predictions (list): List of predicted gloss strings.
        targets (list): List of target (gold) gloss strings.

    Returns:
        dict: A dictionary with keys 'average_accuracy' and 'accuracy'.
    """
    if not targets:
        return {'average_accuracy': 1.0, 'accuracy': 1.0}

    total_correct = 0
    total_tokens = 0
    summed_accuracies = 0.0

    for pred, target in zip(predictions, targets):
        pred_tokens = [tok for tok in re.split(r"\s|-", pred.strip()) if tok]
        target_tokens = [tok for tok in re.split(r"\s|-", target.strip()) if tok]
        entry_correct = 0
        for i in range(len(target_tokens)):
            if i < len(pred_tokens) and pred_tokens[i] == target_tokens[i] and pred_tokens[i] != '<unk>':
                entry_correct += 1
        summed_accuracies += entry_correct / len(target_tokens)
        total_correct += entry_correct
        total_tokens += len(target_tokens)

    return {'average_accuracy': summed_accuracies / len(targets),
            'accuracy': total_correct / total_tokens}