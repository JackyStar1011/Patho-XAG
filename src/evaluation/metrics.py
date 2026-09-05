import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)

def compute_metrics(
    labels,
    probabilities,
    threshold=0.5,
):
    """
    Compute binary classification metrics.

    Args:
        labels:
            Ground-truth labels, shape [N].
            0 = Normal
            1 = Adenoma

        probabilities:
            Probability of class 1 (Adenoma), shape [N].

        threshold:
            Classification threshold.

    Returns:
        Dictionary of metrics.
    """

    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)

    predictions = (probabilities >= threshold).astype(int)

    auroc = roc_auc_score(
        labels,
        probabilities,
    )

    auprc = average_precision_score(
        labels,
        probabilities,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    f1 = f1_score(
        labels,
        predictions,
    )

    sensitivity = recall_score(
        labels,
        predictions,
        pos_label=1,
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = tn / (tn + fp)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": accuracy,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


if __name__ == "__main__":

    labels = [
        0,
        0,
        1,
        1,
    ]

    probabilities = [
        0.1,
        0.4,
        0.6,
        0.9,
    ]

    metrics = compute_metrics(
        labels,
        probabilities,
    )

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")