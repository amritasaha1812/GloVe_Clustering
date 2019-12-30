#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 20:53:55 2019

@author: amrita
"""

import json
import spacy
import numpy as np
nlp = spacy.load('en_core_web_sm')
visual_genome_dir = '../VisualGenome/data/preprocessed/'
concepts_file = 'synsets.txt'
if concepts_file.endswith('.json'):
        read_concepts = json.load(open(visual_genome_dir+concepts_file))
elif concepts_file.endswith('.txt'):
        read_concepts = [x.strip() for x in open(visual_genome_dir+concepts_file).readlines()]
concepts = {}
for x in read_concepts:
        pos_tag = x.split('.')[-2]
        word = '.'.join(x.split('.')[:-2]).replace(' ','_')
        lemma = ' '.join([xi.text if xi.lemma_.startswith('-') else xi.lemma_ for xi in nlp(word.replace('-',' '))])
        concepts[word.replace(' ','_')] = x
        concepts[lemma.replace(' ','_')] = x 

#attributes = set(['.'.join(x.split('.')[:-2]).replace(' ','_') for x in json.load(open('../VisualGenome/data/attribute_types.json'))])
glove = {x.strip().split(' ')[0]:[float(xi) for xi in x.strip().split(' ')[1:]] for x in open('/dccstor/cssblr/amrita/GloVe_Clustering/data/glove/glove.6B.100d.txt').readlines()}
vector_len = len(glove[list(glove.keys())[0]])
glove_extracted = {}
for x,v in concepts.items():
    if '_' in x:
        words = x.split('_')
        embs = np.mean([glove[w] for w in words if w in glove], axis=0)
    elif x in glove:
        embs = glove[x]
    else:
        continue
    if v in glove_extracted:
        glove_extracted[v] = np.mean(np.asarray([embs,  glove_extracted[v]]), axis=0)
    glove_extracted[v] = embs    

len_glove_extracted = len(glove_extracted)
print ('Total number of concepts', len(concepts))
fw=open('/dccstor/cssblr/amrita/GloVe_Clustering/data/glove/glove.VisualGenomeConcepts.'+str(len_glove_extracted)+'.100d.txt','w')
for g in glove_extracted:
        fw.write(g+' '+' '.join([str(x) for x in glove_extracted[g]])+'\n')
fw.close()
