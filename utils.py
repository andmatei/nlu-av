import numpy as np
import pandas as pd
import nltk

from abc import ABC, abstractmethod

class Preprocessor(ABC):
    @abstractmethod
    def map(self, doc):
        pass


class Corpus:

    def __init__(self, file: str):
        self._df = pd.read_csv(file)

        columns = self._df.columns
        self._docs_L = self._df[columns[0]]
        self._docs_R = self._df[columns[1]]
        self._labels = self._df[columns[2]]

    def get_dataset():
        pass

    def preprocess():
        pass

corpus = Corpus("./data/dev.csv")
print(corpus._docs_L)
