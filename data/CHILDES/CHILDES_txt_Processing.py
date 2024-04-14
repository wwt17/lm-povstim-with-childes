#!/usr/bin/env python
# coding: utf-8

# # Childes Procsssing 

# In[1]:

def listify_data(raw_string):
    return [line.split() for line in raw_string.splitlines()]

def read_data(filename):
    with open(filename, "r") as f:
        raw_data = f.read()
    return raw_data


# In[2]:

def unlistify_data(data):
    zipped = [" ".join(line) for line in data]
    return "\n".join(zipped)

def write_data(data, filename):
    with open(filename, "w") as f:
        f.write(unlistify_data(data))


# ## Cleanup
# 
# Remove '\_' and sentences that only contain a single token (sentences that only have '.', or '?', etc.).

# In[3]:


def clean_and_listify(data):
    new_data = [(filename, " ".join(utt).replace("_"," ").split()) for filename, utt in data if len(utt) > 1]
    return new_data


# ## Split possesives and contractions
# 
# Seperate possesives: "camel's" -> "camel 's"
# 
# Also seperate contractions. The subwords _n't_ , _'re_ , _'ll_ , _'m_ , _'ve_ , and _'d_ should be prepened with spaces. 

# In[4]:


# this takes a minute and a half to run on my machine

def split_possesives_and_contractions(word):
    if word.endswith("'s"):
        return word[:-2] + " 's"
    if word == "can't":
        return "can n't"
    if word.endswith("n't"):
        return word[:-3] + " n't"
    if word.endswith("'re"):
        return word[:-3] + " 're"
    if word.endswith("'m"):
        return word[:-2] + " 'm"
    if word.endswith("'d"):
        return word[:-2] + " 'd"
    if word.endswith("'ll"):
        return word[:-3] + " 'll"
    if word.endswith("'ve"):
        return word[:-3] + " 've"
    if word.endswith("s'"):
        return word[:-1] + " '"
    if word.endswith("'r"):
        return word[:-2] + " are"
    if word.endswith("'has"):
        return word[:-4] + " has"
    if word.endswith("'is"):
        return word[:-3] + " is"
    if word.endswith("'did"):
        return word[:-4] + " did"
    if word == "wanna":
        return "want to"
    if word == "hafta":
        return "have to"
    if word == "gonna":
        return "going to"
    if word == "okay":
        return "ok"
    if word == "y'all":
        return "you all"
    if word == "c'mere":
        return "come here"
    if word == "I'ma":
        return "I am going to"
    if word == "what'cha":
        return "what are you"
    if word == "don'tcha":
        return "do you not"
    
    
    # List of startswith exceptions: ["t'", "o'", "O'", "d'"]
    # List of == exceptions: ["Ma'am", "ma'am", "An'", "b'ring", "Hawai'i","don'ting", "rock'n'roll" "don'ting", "That'scop","that'ss","go'ed", "s'pose", "'hey", "me'", "shh'ell", "th'do", "Ross'a", "him'sed"] 
    # List of in exceptions: ["_", "-"]
    # List of endswith exceptions (note that this one is a catch all condition): ["'"]

    return word

def split_line(line):
    s = [split_possesives_and_contractions(word) for word in line]
    return " ".join(s).split()

def split_data(data):
    return [(filename, split_line(line)) for filename, line in data]


# ## Unking
# 
# Replace infrequent words with `<unk>` tokens. 
# 
# Note that the unked tokens are based on the training set, even for the validation and test sets.

# In[5]:


from collections import Counter

def count_frequencies(data):
    frequencies = Counter()
    for filename, line in data:
        frequencies.update(line)
    return frequencies

# words with frequency > cutoff
def make_vocab(data, cutoff=0):
    vocab = count_frequencies(data)
    vocab = sorted(vocab.items(), key = lambda item: (-item[1], item[0]))
    if cutoff:
        vocab = list(filter(lambda item: item[1] > cutoff, vocab))
    return {word: idx for idx, (word, freq) in enumerate(vocab)}

def unk(data, vocab, unk_token="<unk>"):
    unked_data = []
    for filename, line in data:
        unked_line = [(word if word in vocab else unk_token) for word in line]
        unked_data.append((filename, unked_line)) 
    return unked_data


# In[ ]:


def clean_and_unk(dataset, unking=False, unk_token="<unk>", cutoff=0):
    # Clean all the datasets
    dataset = {split: clean_and_listify(data) for split, data in dataset.items()}

    # Split possessives and contractions in all the datasets
    dataset = {split: split_data(data) for split, data in dataset.items()}

    # Create the vocab: All words that occurs more than 2 times
    # in the training set
    vocab = make_vocab(dataset["train"], cutoff=cutoff)

    # unk the train, valid, and test data
    if unking:
        for split in ["train", "valid", "test"]:
            if split in dataset:
                dataset[split] = unk(dataset[split], vocab, unk_token=unk_token)
        vocab = [unk_token] + list(vocab)
    else:
        vocab = list(vocab)

    return dataset, vocab

