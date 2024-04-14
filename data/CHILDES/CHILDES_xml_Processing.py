#!/usr/bin/env python
# coding: utf-8

# Instructions to read the XML version of the CHILDES corpus adapted from on the [nltk website](http://www.nltk.org/howto/childes.html). 
# 
# XML corpora can be downloaded from the [childes website](https://childes.talkbank.org/data-xml/Eng-NA/)

# In[1]:


import nltk
from childes import CHILDESCorpusReader # Edited version of nltk.corpus.reader
from collections import defaultdict
import random
import copy
import shutil  
import os
from os import path
import fnmatch

def copy_directory(source, destination):

    # Copy the content of
    # source to destination
    destination = shutil.copytree(source, destination)


def find_replace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)

def preprocess(path_to_corpora, corpora_file_name):
    source = path_to_corpora + corpora_file_name
    destination = source +  "-preprocessed"
    if path.exists(destination):
        print("Preprocessed directory " + destination + 
                " already exists. Remove directory and run this script again to re-preprocess")
        return
    copy_directory(source, destination)
    find_replace(destination, "<g>", "", "*.xml")
    find_replace(destination, "</g>", "", "*.xml")
    find_replace(destination, '<p type="drawl"/>', "", "*.xml")
    find_replace(destination, "<shortening>", "", "*.xml")
    find_replace(destination, "</shortening>", "", "*.xml")

# print(destination) prints the
# path of newly created file

# Create a CHILDESCorpusReader object
def read_corpora(path_to_corpora, corpora_file_name):
    return CHILDESCorpusReader(path_to_corpora, corpora_file_name + "/.*.xml")


# Input: CHILDESCorpusReader object
# Output: A dict whose keys are the fileids from the
#         CHILDESCorpusReader's corpus, and whose values
#         are a list of utterances in that file made by
#         any participant except the target child
def map_files_to_non_target_child_utterances(corpora):
    filtered_corpora = {}
    for fileid in corpora.fileids():
        participants = get_non_target_child_participants(corpora, fileid)
        utterances = get_utterances_filtered_by_participants(corpora, fileid, participants)
        if utterances != []:
            filtered_corpora[fileid] = utterances
    return filtered_corpora

# Returns list of participant IDs for participants who
# are not the target child
def get_non_target_child_participants(corpora, fileid):
    target_child_coded_as_Target_Child = False
    participants_coded_as_Child = []
    non_target_child_participants = []
    corpora_participants = corpora.participants(fileid)
    for participants in corpora_participants:
        for key in participants.keys():
            dct = participants[key]
            if dct['role'] not in ["Target_Child","Child"]:
                non_target_child_participants.append(dct['id'])
            if dct['role'] == "Target_Child":
                target_child_coded_as_Target_Child = True
            if dct['role'] == "Child":
                participants_coded_as_Child.append(dct['id'])
    if target_child_coded_as_Target_Child:
        non_target_child_participants += participants_coded_as_Child
    return non_target_child_participants

# Returns utterances from `fileid` in `corpus` spoken by any
# participant in `participants`
def get_utterances_filtered_by_participants(corpus, fileid, participants):
    utterances = corpus.sents(fileid, speaker=participants, replace=True) # replace=True
    cleaned_utts = [utt for utt in utterances if utt != []]
    return cleaned_utts


# Checks if fileid is in the treebank
def is_treebank_file(fileid):
    for treebank_corpus_name in ['Brown','Soderstrom','Valian','Suppes','HSLLD/HV1']:
        if treebank_corpus_name in fileid:
            return True
    return False

# Split all the utterances into those that are
# in the treebank and not in the treebank
def split_treebank(files_to_utterances):
    treebank = {file: utterances for file, utterances in files_to_utterances.items() if is_treebank_file(file)}
    not_treebank = {file: utterances for file, utterances in files_to_utterances.items() if not is_treebank_file(file)}
    return treebank, not_treebank

# Input: List of sentences
# Output: Number of questions in the list 
def count_questions(sents):
    return len([sent for sent in sents if sent[-1] == '?'])

# Input: Dict whose keys are fileids and values are lists of utterances
#        in the file
# Output: Same dict but sorted by the number of questions in the file
def sort_dict_by_number_of_questions(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: count_questions(item[1]))}

# Given a files_to_utterances dictionary, split into 2 splits:
# an `included` split containing 90% of the data, and `excluded` 
# containing 10% of the data.
# Works by sorting the files by number of questions and then excluding 
# every 10th file so that the excluded contains only entire files and 
# contains approximately 10% of the questions
def hold_out(files_to_utterances, exclude_every_kth=10):
    files_to_utterances_sorted = sort_dict_by_number_of_questions(files_to_utterances)
    included = copy.deepcopy(files_to_utterances_sorted)
    excluded = {}
    for i,file in enumerate(files_to_utterances_sorted):
        if i % exclude_every_kth == 0:
            excluded[file] = included.pop(file)
    return included, excluded


# Split a files_to_utterances dict into training, validation,
# and test splits.
def split(
        files_to_utterances,
        split_ratio={"valid": 5, "test": 5, "train": 90},
        shuffling=True,
):
    files_to_utterances_sorted = sort_dict_by_value_length(files_to_utterances)
    utterances = [(filename, utts) for filename, utts in files_to_utterances_sorted.items()]
    dataset_dict = {split: [] for split in split_ratio}
    batch_size = sum(split_ratio.values())
    count = 0
    while count < len(utterances):
        batch_size_ = min(batch_size, len(utterances) - count)
        indices = list(range(count, count + batch_size_))
        if shuffling:
            random.shuffle(indices)
        count_ = 0
        for split, ratio in split_ratio.items():
            for i in indices[count_:count_+ratio]:
                filename, utts = utterances[i]
                dataset_dict[split].extend((filename, utt) for utt in utts)
            count_ += ratio
        count += batch_size_
    return dataset_dict

# Sort a dictionary in descending order by the lengths of its values
def sort_dict_by_value_length(x):
    return {k: v for k, v in sorted(x.items(), key=lambda item: len(item[1]), reverse=True)}


# Reshuffle the validation and test data to add the treebank test data
# (aka `excluded`) to the test set, and then move some elements from `test`
# into `valid` so that valid and test are still the same size
# we also make sure not to split files up between test and validation
def remix_held_out(valid, test, excluded):
    excluded_utterances = [(filename,utt) for filename,utts in excluded.items() for utt in utts]
    excluded_size = len(excluded_utterances)
    reshuffle_size = int(excluded_size/2)
    if excluded_size > 0:
        cutoff_file = test[-reshuffle_size][0]
        while cutoff_file == test[-reshuffle_size][0]: 
            reshuffle_size -= 1
    return valid + test[-reshuffle_size:], test[:-reshuffle_size] + excluded_utterances


def data_to_files_to_utterances(data):
    files_to_utterances = {}
    for f,u in data:
        if f in files_to_utterances:
            files_to_utterances[f].append(u)
        else:
            files_to_utterances[f] = [u]
    return files_to_utterances


def files_to_utterances_to_data(file_to_utterances):
    if isinstance(file_to_utterances, dict):
        items = file_to_utterances.items()
    else:
        items = file_to_utterances
    return [(f, utt) for f, utts in items for utt in utts]


def shuffle(data, shuffling=False):
    files_to_utterances = data_to_files_to_utterances(data)
    files_to_utterances_list = list(files_to_utterances.items())
    if shuffling:
        random.shuffle(files_to_utterances_list)
    return files_to_utterances_to_data(files_to_utterances_list)
    

def process_childes_xml(
        path_to_childes="./",
        childes_file_name="childes-xml",
        splitting=True,
        shuffling=False,
        seed=1,
):
    if not shuffling:
        print("REMEMBER THAT YOU TURNED OFF SHUFFLING!! turn it back on when you are done with checking out the results")
    random.seed(seed)

    # Preprocessing (removes <g> tags)
    print("Starting preprocessing")
    preprocess(path_to_childes, childes_file_name)
    print("Preprocessing finished")

    # Create corpus reader
    corpora = read_corpora(path_to_corpora=path_to_childes, corpora_file_name=childes_file_name + "-preprocessed")
    
    # Get utterances from all participants other than target child
    files_to_utterances = map_files_to_non_target_child_utterances(corpora)

    if not splitting:
        return {"train": files_to_utterances_to_data(files_to_utterances)}

    # Split the utterances into those from the treebank and
    # not from the treebank
    treebank, not_treebank = split_treebank(files_to_utterances)

    # Split the treebank into included and excluded splits
    included_treebank, excluded = hold_out(treebank)

    # The full set of items included in pretraining is those
    # that are not in the treebank and those that are in the
    # included_treebank split
    included = {**not_treebank, **included_treebank} # Python 3.5 or greater

    # Split the pretraining data into train, valid, and test
    dataset_dict = split(included)

    # Add the excluded treebank data to the second half of `test` to make the
    # final test set, and then combind `valid` with the first half of `test` to
    # make the final validation set
    valid_remixed, test_remixed = remix_held_out(
        dataset_dict["valid"],
        dataset_dict["test"],
        excluded)

    # The full list of excluded treebank utterances
    excluded = files_to_utterances_to_data(excluded)

    dataset_dict = {
        "train": dataset_dict["train"],
        "valid": valid_remixed,
        "test": test_remixed,
        "excluded": excluded,
    }
    dataset_dict = {
        split: shuffle(data, shuffling=shuffling)
        for split, data in dataset_dict.items()}
    return dataset_dict

