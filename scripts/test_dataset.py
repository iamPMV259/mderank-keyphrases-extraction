import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.keyphrase_extractor import extract_keyphrases


def calculate_f1(predicted, ground_truth, k):
    """
    Calculate precision, recall, and F1@K.
    
    Parameters:
      predicted (list): List of predicted keyphrases.
      ground_truth (list): List of ground truth keyphrases.
      k (int): The cutoff for evaluation.
    
    Returns:
      tuple: precision, recall, and F1 score.
    """
    predicted_top_k = predicted[:k]
    common = set(predicted_top_k) & set(ground_truth)
    precision = len(common) / k if k > 0 else 0
    recall = len(common) / len(ground_truth) if ground_truth else 0
    f1 = 0
    if precision + recall > 0:
       f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1



    