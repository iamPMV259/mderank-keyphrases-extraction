import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.get_dataset import *
from src.keyphrase_extractor import extract_keyphrases


def calculate_f1(predicted, ground_truth, k)->float:
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
    precision = len(common) * 1.0 / k if k > 0 else 0
    recall = len(common) * 1.0 / len(ground_truth) if ground_truth else 0
    f1 = 0
    if precision + recall > 0:
       f1 = 200.0 * precision * recall / (precision + recall)
    # return precision, recall, f1
    return f1


def print_to_json(data_name, k, score):
    """
    Print the evaluation results to a JSON file.
    
    Parameters:
      data_name (str): The name of the dataset.
      k (int): The cutoff for evaluation.
      score (list): The list of evaluation results.
    """
    average_score = sum(score) / len(score)
    result = {
        "dataset": data_name,
        "top_k": k,
        "average_score": average_score,}
    if not os.path.exists("results"):
        os.makedirs("results")
    if not os.path.exists("results/{data_name}_{k}.json"):
        os.touch(f"results/{data_name}_{k}.json")
    with open(f"results/{data_name}_{k}.json", "w") as f:
        json.dump(result, f)
    


if __name__ == "__main__":
    dataset = ['duc2001', 'inspec', 'krapivin', 'nus', 'semeval2010', 'sameval2017']

    for data_name in dataset:
        if data_name == "duc2001":
            data, labels = get_duc2001_data()
        elif data_name == "inspec":
            data, labels = get_inspec_data()
        elif data_name == "krapivin":
            data, labels = get_krapivin_data()
        elif data_name == "nus":  
            data, labels = get_nus_data()
        elif data_name == "semeval2010":
            data, labels = get_semeval2010_data()
        elif data_name == "sameval2017":
            data, labels = get_semeval2017_data()
        
        for k in [5, 10, 15]:
            score = []
            for id in data:
                keyphrases = extract_keyphrases(data[id], top_k=k)
                f1 = calculate_f1(keyphrases, labels[id], k)
                score.append(f1)
                # print(id, " ==> ", f1)
            print_to_json(data_name, k, score)


    