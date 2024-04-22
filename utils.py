
import spacy
import json
import numpy as np
import pandas as pd
import nltk
import textacy.preprocessing as tp
import fasttext
import fasttext.util
import torch

from keras.preprocessing.sequence import pad_sequences
from datasets import Dataset
from gensim.models import FastText

from abc import ABC, abstractmethod
from enum import Enum


class DataMapper(ABC):
    @abstractmethod
    def map(self, text, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.map(*args, **kwargs)


class POSTagger(DataMapper):
    def __init__(self, corpora):
        self._corpora = corpora

    def map(self, text):
        return nltk.pos_tag(text)
    

class WordGramMapper(DataMapper):
    def __init__(self):
        pass

    def map(self, text, n=2):
        return list(nltk.ngrams(text, n))
    
    
class CharacterGramMapper(DataMapper):
    def __init__(self):
        pass

    def map(self, text, n=3):
        chars = [char for word in text for char in word]
        return list(nltk.ngrams(chars, n))
    

class TextDistorter(DataMapper):
    def __init__(self, corpora):
        self._corpora = corpora
        self._wordfreq = nltk.FreqDist(
            [word.lower() for text in self._corpora for word in text]
        )

    def map(self, text, k, multiple=False, char="*", digit="#"):
        word_set = [w[0] for w in self._wordfreq.most_common(k)]

        for word, i in zip(text, range(len(text))):
            if word.lower() not in word_set:
                text[i] = self._encode(word, multiple, char, digit)

        return text

    def _encode(self, word, multiple=False, char="*", digit="#"):
        result = ""

        char_found = False
        digit_found = False

        for c in word:
            if c.isalpha():
                if multiple or (not char_found and  multiple):
                    result += char
                    char_found = True
                    digit_found = False
            elif c.isdigit():
                if multiple or (not digit_found and not multiple):
                    result += digit
                    digit_found = True
                    char_found = False
            else:
                result += c
                char_found = False
                digit_found = False

        return result
    

class MappingPipeline:
    def __init__(self):
        self._mappers = []

    def add(self, mapper: DataMapper):
        self._mappers.append(mapper)

    def process(self, text):
        for mapper in self._mappers:
            text = mapper.map(text)

        return text
    
    def __call__(self, text):
        return self.process(text)
    

class Vocabulary:
    def __init__(self, vocab_size=10000, cutoff=0, unk_token="<UNK>"):
        self._vocab_size = vocab_size
        self._unk_token = unk_token
        self._cutoff = cutoff

        self._filtered = False
        self._vocab = {
            self._unk_token: 0
        }
        self._counts = {
            self._unk_token: self._cutoff
        }

    def add(self, doc):
        self._filtered = False
        for token in doc:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                self._counts[token] = 0
            self._counts[token] += 1

    def __getitem__(self, token):
        if token in self._vocab and self._counts[token] > self._cutoff:
            return self._vocab[token]
        return self._vocab[self._unk_token]

    def __len__(self):
        return len(self._vocab)
    
    def __iter__(self):
        if not self._filtered:
            self._filtered = True
            q = sorted(self._counts.items(), key=lambda x: x[1], reverse=True)
            self._filtered_vocab = list(list(zip(*q))[0])[:self._vocab_size]
        
        return iter((token, self._vocab[token]) for token in self._filtered_vocab)
        

class Embeddings(ABC):
    @abstractmethod
    def __getitem__(self, token):
        pass

    @abstractmethod
    def get_weights_tensor(self):
        pass


class CustomFastTextEmbeddings(Embeddings):
    def __init__(self, vocab_size=10000, embedding_size = 300):
        self._vocab_size = vocab_size
        self._embedding_size = embedding_size

        self._model = FastText(
            vector_size=self._embedding_size, 
            window=5, 
            max_vocab_size=vocab_size,
            min_count=1, 
            workers=4, 
            sg=1
        )

    @staticmethod
    def load(input):
        ft = CustomFastTextEmbeddings()
        ft._model = FastText.load(input)
        return ft

    def train(self, corpus, output=None):
        self._model.build_vocab(corpus)
        self._model.train(corpus, total_examples=self._model.corpus_count, epochs=10)
        if output is not None:
            self._model.save(output)

    def __getitem__(self, token):
        return self._model[token]
    
    def get_weights_tensor(self):
        weights = torch.zeros(self._model.wv.vectors.shape[0], self._model.wv.vectors.shape[1])
        for token, i in self._model.wv.key_to_index.items():
            weights[i] = self[token]
        return weights


class FastTextEmbeddings(Embeddings):
    def __init__(self):
        fasttext.util.download_model('en', if_exists='ignore')
        self._model = fasttext.load_model('cc.en.300.bin')

    def __getitem__(self, token):
        return self._model.get_word_vector(token)
    
    def get_weights_tensor(self):
        weights = torch.zeros(self._model.get_output_matrix().shape)
        for token, i in self._model.get_words(include_freq=False):
            weights[i] = self[token]
        return weights


class Corpus:
    def __init__(self, max_sent_len=50, max_doc_len=50, vocab_size=10000):
        self._vocab_size = vocab_size
        self._max_sent_len = max_sent_len
        self._max_doc_len = max_doc_len
        
        self._tokenizer = spacy.load("en_core_web_lg")

        self._word_vocab = Vocabulary(vocab_size=self._vocab_size)
        self._char_vocab = Vocabulary(vocab_size=self._vocab_size)

        self._docs_L = []
        self._docs_R = []
        self._labels = []
        
    def save(self, file:str):
        l = [
            {
                "doc_L": self._docs_L[i],
                "doc_R": self._docs_R[i],
                "label": self._labels[i]
            }
            for i in range(len(self._docs_L))
        ]

        json.dump(l, open(file, "w"))

    def open(self, file:str, preprocessed=False):
        if preprocessed:
            data = json.load(open(file, "r"))
            
            for item in data:
                self._docs_L.append(item["doc_L"])
                self._docs_R.append(item["doc_R"])
                self._labels.append(item["label"])

            return True

        self._df = pd.read_csv(file)

        columns = self._df.columns
        df_docs_L = self._df[columns[0]]
        df_docs_R = self._df[columns[1]]
        df_labels = self._df[columns[2]]

        self._docs_L = []
        for i, doc in enumerate(df_docs_L):
            if not isinstance(doc, str):
                doc = ""
            doc = self._preprocess_doc(doc)
            doc = self._pad(doc)
            self._docs_L.append(doc)
        
        self._docs_R = []
        for i, doc in enumerate(df_docs_R):
            if not isinstance(doc, str):
                doc = ""
            doc = self._preprocess_doc(doc)
            doc = self._pad(doc)
            self._docs_R.append(doc)
        
        self._labels = df_labels.tolist()

        return True
    
    def split(self, ratio=0.8):
        n = len(self._docs_L)
        m = int(n * ratio)

        train = Corpus()
        train._docs_L = self._docs_L[:m]
        train._docs_R = self._docs_R[:m]
        train._labels = self._labels[:m]

        test = Corpus()
        test._docs_L = self._docs_L[m:]
        test._docs_R = self._docs_R[m:]
        test._labels = self._labels[m:]

        return train, test
    
    @property
    def dataset(self):
        return Dataset.from_dict({
            "doc_L": self._docs_L,
            "doc_R": self._docs_R,
            "label": self._labels
        })
    
    def get_all_docs(self):
        return self._docs_L + self._docs_R

    def _preprocess_doc(self, doc):
        doc = tp.normalize.whitespace(doc)
        doc = tp.normalize.quotation_marks(doc)
        doc = tp.normalize.unicode(doc)

        doc = self._tokenizer(doc)
        doc = [[token.text for token in sent] for sent in doc.sents]

        return doc     

    def _pad(self, doc):
        return pad_sequences(
            doc, 
            maxlen=self._max_sent_len, 
            padding="post",
            truncating="post",
            dtype=object,
            value="<PAD>"
        ).tolist()

    def _add_special_tokens(self, doc):
        result = []
        
        for sent in doc:
            sent = ['<SOS>'] + sent
            if len(sent) < self._max_sent_len:
                sent = sent + ['<PAD>'] * (self._max_sent_len - len(sent) - 1) + ['<EOS>']
                result.append(sent)
            else:
                while(len(sent) > 1):
                    if len(sent) < self._max_sent_len:
                        sent = sent + ['<PAD>'] * (self._max_sent_len - len(sent) - 1) + ['<EOS>']
                        result.append(sent)
                    else:
                        sent = sent[:self._max_sent_len - 1] + ['<ELB>']
                        result.append(sent)
                    sent = ['<SLB>'] + sent[self._max_sent_len - 1:]

        if len(result) < self._max_doc_len:
            result = result + [['<PAD>'] * self._max_sent_len] * (self._max_doc_len - len(result))

        return result

    def _make_vocabularies(self, docs):
        for doc in docs:
            for sent in doc:
                self._word_vocab.add(sent)
                for token in sent:
                    self._char_vocab.add(token)


# ft = FastTextEmbeddings.load()
# print(ft["hello"])
# print(ft[","])
# print(ft["<UNK>"])

# corpus = Corpus()
# corpus.open("data/dev.csv")
# print(corpus._docs_L[0])
# print(type(corpus._docs_L))
# corpus.save("data/dev.json")