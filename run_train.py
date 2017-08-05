import os
from data_sampler import DataSampler
from binary_classification import BinaryClassifier

def main():
    ds = DataSampler()
    classifier = BinaryClassifier(hidden_size=64, data_sampler=ds)
    classifier.train()

if __name__ == '__main__':
    main()