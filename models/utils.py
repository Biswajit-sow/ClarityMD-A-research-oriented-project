import torch
from collections import Counter

def calculate_class_weights(dataset):
    """
    Calculates class weights for a dataset to handle imbalance.
    The formula is: weight = total_samples / (num_classes * num_samples_per_class)

    Args:
        dataset (torch.utils.data.Dataset): The dataset object. It must have a 'labels' attribute.

    Returns:
        torch.Tensor: A tensor of weights for each class.
    """
    if not hasattr(dataset, 'labels'):
        raise AttributeError("Dataset must have a 'labels' attribute to calculate class weights.")
    
    labels = dataset.labels
    class_counts = Counter(labels)
    num_samples = len(labels)
    num_classes = len(class_counts)

    weights = [num_samples / (num_classes * class_counts[i]) for i in sorted(class_counts.keys())]
    
    print(f"Calculated class weights: {weights}")
    return torch.tensor(weights, dtype=torch.float)