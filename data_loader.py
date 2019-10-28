#!/usr/bin/env python
# coding: utf-8

# In[57]:


import warnings; warnings.filterwarnings('ignore')
from os import walk
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader


# In[ ]:


class WordDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# In[73]:


class MRLoader:
    def __init__(self, batch_size=64):
        self.batch_size=batch_size
        
        self.pos_data_path = "/home/hdd/txt_sentoken/pos/"
        self.neg_data_path = "/home/hdd/txt_sentoken/neg/"
        self.pos_dataset = []
        self.neg_dataset = []
        
        self.tokenizer = Tokenizer()

        for (dirpath, dirnames, pos_filenames) in walk(self.pos_data_path): break
        for (dirpath, dirnames, neg_filenames) in walk(self.neg_data_path): break

        for filename in pos_filenames:
            with open(self.pos_data_path + filename) as test_file:
                self.pos_dataset.append("".join(test_file.readlines()))

        for filename in neg_filenames:
            with open(self.neg_data_path + filename) as test_file:
                self.neg_dataset.append("".join(test_file.readlines()))

        self.pos_df = pd.DataFrame({"text": self.pos_dataset, "label": [1] * len(self.pos_dataset)})
        self.neg_df = pd.DataFrame({"text": self.neg_dataset, "label": [0] * len(self.neg_dataset)})
        
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4
        self.X_train_df, self.X_test_df, self.y_train_df, self.y_test_df = [None] * 4
        self.train_loader, self.test_loader = [None] * 2
        
        self.already_splitted = False

    def get_dataset(self, random_state=32):
        if not self.already_splitted:
            X_pos_train, X_pos_test, y_pos_train, y_pos_test = train_test_split(
                self.pos_df[['text']], self.pos_df[['label']],
                random_state=random_state
            )
            X_neg_train, X_neg_test, y_neg_train, y_neg_test = train_test_split(
                self.neg_df[['text']], self.neg_df[['label']],
                random_state=random_state
            )

            self.X_train_df, self.X_test_df = pd.concat([X_pos_train, X_neg_train]), pd.concat([X_pos_test, X_neg_test])
            self.y_train_df, self.y_test_df = pd.concat([y_pos_train, y_neg_train]), pd.concat([y_pos_test, y_neg_test])

            self.tokenizer.fit_on_texts(self.X_train_df['text'])

            train_sequence = self.tokenizer.texts_to_sequences(self.X_train_df['text'])
            train_sequence = sequence.pad_sequences(train_sequence, maxlen=300)
            test_sequence = self.tokenizer.texts_to_sequences(self.X_test_df['text'])
            test_sequence = sequence.pad_sequences(test_sequence, maxlen=300)

            train_indices = np.arange(train_sequence.shape[0])
            test_indices = np.arange(test_sequence.shape[0])
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

            self.X_train, self.X_test = torch.FloatTensor(train_sequence[train_indices]), torch.FloatTensor(test_sequence[test_indices])
            self.y_train, self.y_test = torch.FloatTensor(self.y_train_df.values[train_indices]), torch.FloatTensor(self.y_test_df.values[test_indices])
            
            train = WordDataset(self.X_train, self.y_train)
            test = WordDataset(self.X_test, self.y_test)
            self.train_loader = DataLoader(train, batch_size = self.batch_size, shuffle=True)
            self.test_loader = DataLoader(test, batch_size = self.batch_size, shuffle=True)
            
            self.already_splitted = True
        return self.train_loader, self.test_loader
    def get_train_text(self):
        return self.X_train_df


# In[ ]:


class Glove:
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        
        self.embeddings_index = dict()
        with open('/home/hdd/glove.6B.300d.txt', encoding="utf8") as infile:
            for line in infile:
                word_vector = line.split()
                word = word_vector[0]
                coefs = np.asarray(word_vector[1:], dtype='float32')
                self.embeddings_index[word] = torch.tensor(coefs)
                
        self.vocab_size = len(self.embeddings_index.keys())
        self.embedding_matrix = torch.zeros((self.vocab_size, self.embedding_dim))
    
    def get_embedding(self, tokenizer):
        self.tokenizer = tokenizer
        
        for word, index in tokenizer.word_index.items():
            if index > self.vocab_size - 1:
                break
            else:
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    self.embedding_matrix[index] = embedding_vector
        
        return self.embedding_matrix

