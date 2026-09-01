import os, shutil
from os import listdir
from os.path import isfile, join
import string
import random
import nltk
import re
import numpy as np
import heapq

def list_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]

def get_filename(dir, fname):
    return os.path.join(dir, fname)

def get_filenames(source_dir):
    fnames = []
    labels = []
    for fname in list_files(source_dir):
        fnames.append(get_filename(source_dir, fname))
        labels.append(fname.split(".",1)[0])
    return fnames, labels

# Read a file into a String
def read_file_to_string(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read().replace('\n', ' ')
            return data
    except:
        return ""

def read_file(file_name, word_count):
    texts = []

    text = read_file_to_string(file_name)

    dataset = nltk.sent_tokenize(text)
    for i in range(len(dataset)):
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W', ' ', dataset[i])
        dataset[i] = re.sub(r'\s+', ' ', dataset[i])
    texts.append(dataset)

    for data in dataset:
        words = nltk.word_tokenize(data)
        for word in words:
            if word not in word_count.keys(): word_count[word] = 1
            else: word_count[word] += 1

    return texts, word_count

def get_freq_words(dsize, wc):
    return heapq.nlargest(dsize, wc, key=wc.get)

def get_vector(intext, freq_words):
    ds = nltk.sent_tokenize(intext)
    for i in range(len(ds)):
        ds[i] = ds[i].lower()
        ds[i] = re.sub(r'\W', ' ', ds[i])
        ds[i] = re.sub(r'\s+', ' ', ds[i])

    retval = []
    for data in ds:
        vector = []
        for word in freq_words:
            if word in nltk.word_tokenize(data):
                vector.append(1)
            else:
                vector.append(0)
        retval.append(vector)
    return retval
