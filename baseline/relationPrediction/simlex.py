import pandas as pd
import numpy as np
from collections import defaultdict
from nltk.corpus import wordnet as wn
import torch
import torch.nn as nn
from scipy.stats import pearsonr


class SimLex999():
    def __init__(self, data_path='datasets/SimLex-999.txt'):
        self.data = pd.read_csv(data_path, sep='\t', header=0)
    
    def eval(self, mapper):
        invalid_idx = []
        model_sim = np.zeros(999)
        ground_truth_sim = self.data['SimLex999']

        for i, row in self.data.iterrows():
            w1 = row['word1']
            w2 = row['word2']
            sim = mapper.sim(w1, w2)
            if sim:
                model_sim[i] = sim
            else:
                invalid_idx.append(i)

        model_sim[invalid_idx] = sum(model_sim) / (999-len(invalid_idx))

        return pearsonr(self.rank(model_sim), self.rank(ground_truth_sim))
        
    
    def rank(self, array):
        array = np.array(array)
        order = array.argsort()
        ranks = order.argsort()
        return ranks


class Mapper():
    def __init__(self, entity2id, ent_embeddings, word_level=False):
        if word_level:
            word2embedding = self._get_word2embedding(entity2id, ent_embeddings)
        else:
            word2embedding = self._get_word2embedding_from_synsets(entity2id, ent_embeddings)

        self.w2id = {}
        self.id2w = {}
        self.word_embeddings = torch.zeros((len(word2embedding), len(ent_embeddings[0])))

        for i, (w, e) in enumerate(word2embedding.items()):
            self.w2id[w] = i
            self.id2w[i] = w
            self.word_embeddings[i] = e

    def _get_word2embedding_from_synsets(self, entity2id, ent_embeddings):
        word2ids = defaultdict(list)
        for entity, id in entity2id.items():
            synset = wn.synset(entity)
            words = [sense.name() for sense in synset.lemmas()]
            for w in words:
                word2ids[w].append(id)
        self.word2ids = word2ids
        word2embedding = {}
        for word, ids in word2ids.items():
            word2embedding[word] = torch.mean(ent_embeddings[ids], dim=0)
        
        return word2embedding
    
    def _get_word2embedding(self, entity2id, ent_embeddings):
        word2embedding = {}
        for word, id in entity2id.items():
            word2embedding[word] = ent_embeddings[id]
        
        return word2embedding

    def __getitem__(self, key):
        if isinstance(key, str):
            key = self.w2id[key]
        return self.word_embeddings[key]

    def __len__(self):
        return len(self.word_embeddings)
    
    def __contains__(self, key):
        return key in self.w2id
    
    def sim(self, w1, w2):
        if w1 not in self.word2embedding or w2 not in self.word2embedding:
            return None

        cossim = nn.CosineSimilarity(dim=0)
        return cossim(w1, w2)
    
