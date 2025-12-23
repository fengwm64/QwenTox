import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def loadCSV():
    train_path = r"../../dataset/train.csv"
    test_path = r"../../dataset/test.csv"
    test_labels_path = r"../../dataset/test_labels.csv"
    sample_path = r"../../dataset/sample_submission.csv"
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    test_labels_data = pd.read_csv(test_labels_path)
    sample_data = pd.read_csv(sample_path)
    return train_data,test_data,test_labels_data,sample_data

if __name__ =="__main__":
    pass