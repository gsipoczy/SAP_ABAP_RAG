import os, shutil
from os import listdir
from os.path import isfile, join
import string
import random
import nltk
import re
import numpy as np
import heapq

def get_source_dir(sd):
    return os.path.join(os.getcwd(), sd)

def get_target_dir(td):
    return os.path.join(os.getcwd(), td)

def get_target_train_dir(td):
    return os.path.join(os.getcwd(), td, 'train')

def get_target_test_dir(td):
    return os.path.join(os.getcwd(), td, 'test')

# List of all files in a directory
def list_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]

# Get file name
def get_filename(dir, fname):
    return os.path.join(dir, fname)

# Read a file into a String
def read_file_to_string(filename):
    try:
        with open(filename, 'r') as file:
            data = file.read().replace('\n', ' ')
            return data
    except:
        return ""

# CLEANSING
def cleanse(text):
    words = text.split()
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in words]
    return stripped

# Smallest split
def smallest_split(textlen, multiplier = 1):
    try:
        ssp = 1
        for i in range(20):
            ssp = ssp * (i + 1)
            if ssp >= textlen:
                return (i + 1) * multiplier
    except:
        return 0

def split_text(text_as_list, multiplier = 1):
    txtlen = len(text_as_list)
    txtsplit = smallest_split(txtlen, multiplier)
    txtdiv = txtlen//txtsplit

    main_list = []
    act_list = []

    counter = 0
    for word in text_as_list:
        counter = counter + 1
        if counter > txtdiv:
            main_list.append(act_list)
            act_list = []
            counter = 0
        act_list.append(word)
    return main_list

def flat_it(multi_list):
    try:
        retval = []
        for l1 in multi_list:
            for l2 in l1:
                retval.append(l2)
        return retval
    except:
        return []

def scrumble_up(text_list):
    shuffled = random.sample(text_list, len(text_list))
    retval = []
    for i in shuffled:
        for j in i:
            retval.append(j)
    return retval

def list_to_string(txtlist):
    return ' '.join(txtlist)

def get_a_lot(text, num, multi):
    cln = cleanse(text)
    retval = []
    # 1st text is the original
    retval.append(text)
    # Then go for scrumbles
    for i in range(num):
        txt = split_text(cln, multi)
        sctxt = list_to_string(scrumble_up(txt))
        retval.append(sctxt)
    return retval

# Create directory
def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# Delete files
def empty_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def get_folder_of_text(tdir, which, textname):
    name = textname.split(".",1)[0]
    if which == "TRAIN":
        folder = get_target_train_dir(tdir)
    else:
        folder = get_target_test_dir(tdir)

    folder = os.path.join(folder, name)
    return folder

# Prepare target folders
def prepare_folders(tdir, sdir):
    train_folder = get_target_train_dir(tdir)
    test_folder = get_target_test_dir(tdir)
    create_folder(train_folder)
    create_folder(test_folder)
    empty_folder(train_folder)
    empty_folder(test_folder)

    # Now read the source files and make a folder for all
    for fname in list_files(get_source_dir(sdir)):
        create_folder(get_folder_of_text(tdir, "TRAIN", fname))
        create_folder(get_folder_of_text(tdir, "TEST", fname))


# Write text to file
def write_file(text, fname):
    try:
        with open(fname, 'w') as f:
            f.write(text)
    except:
        pass

# Next file name
def get_file_name(tdir, counter):
    fname = "f" + str(counter) + ".txt"
    return os.path.join(tdir, fname)

# Putting together
def do_it(sdir, tdir, splitm, faketrain, faketest):

    # Prepare the taget 3folders
    prepare_folders(tdir, sdir)

    # General file name counter
    cc = 0

    # Process the source files one by one
    for fname in list_files(get_source_dir(sdir)):

        # Get the source content
        text = read_file_to_string(get_filename(get_source_dir(sdir), fname))

        # TRAIN
        targetdir = get_folder_of_text(tdir, "TRAIN", fname)
        multxt = get_a_lot(text, faketrain, splitm)
        for txt in multxt:
            outfile = get_file_name(targetdir, cc)
            write_file(txt, outfile)
            cc = cc + 1

        # TEST
        targetdir = get_folder_of_text(tdir, "TEST", fname)
        multxt = get_a_lot(text, faketrain, splitm)
        for txt in multxt:
            outfile = get_file_name(targetdir, cc)
            write_file(txt, outfile)
            cc = cc + 1


def get_data(sdir):
    word2count = {}
    texts = []
    abaps = []

    for fname in list_files(sdir):
        text = read_file_to_string(get_filename(sdir, fname))
        abaps.append(fname.split(".",1)[0])

        dataset = nltk.sent_tokenize(text)
        for i in range(len(dataset)):
            dataset[i] = dataset[i].lower()
            dataset[i] = re.sub(r'\W', ' ', dataset[i])
            dataset[i] = re.sub(r'\s+', ' ', dataset[i])
        texts.append(dataset)
        for data in dataset:
            words = nltk.word_tokenize(data)
            for word in words:
                if word not in word2count.keys():
                    word2count[word] = 1
                else:
                    word2count[word] += 1

    return word2count, texts, abaps

def get_freq_words(dsize, wc):
    return heapq.nlargest(dsize, wc, key=wc.get)

def get_bigx(texts, freq_words):
    bigx = []
    for dataset in texts:
        smallx = []
        for data in dataset:
            vector = []
            for word in freq_words:
                if word in nltk.word_tokenize(data):
                    vector.append(1)
                else:
                    vector.append(0)
            smallx.append(vector)
        bigx.append(smallx)
    return bigx

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

def bestmatch(intext, bigx, abaps, freq_words, maxn):
    ds = nltk.sent_tokenize(intext)
    for i in range(len(ds)):
        ds[i] = ds[i].lower()
        ds[i] = re.sub(r'\W', ' ', ds[i])
        ds[i] = re.sub(r'\s+', ' ', ds[i])

    for data in ds:
        vector = []
        for word in freq_words:
            if word in nltk.word_tokenize(data):
                vector.append(1)
            else:
                vector.append(0)

    retval = {}
    idx = 0
    for t in bigx:
        counter = 0
        for k in t:
            try:
                for jidx, j in enumerate(k):
                    if j == 1 and vector[jidx] == 1:
                        counter += 1
            except:
                print("jujuju")
        if counter > 0:
            retval[abaps[idx]] = counter
        idx += 1

    retval = dict(sorted(retval.items(), key=lambda item: item[1], reverse=True))
    retval = dict(list(retval.items())[0: maxn])
    return retval
