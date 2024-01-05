import pandas as pd
from itertools import chain
import numpy as np
import tensorflow as tf
import os
import json

from data_dictionaries import upos_dictionary, deprel_dictionary, arcs_dictionary
from oracle import Oracle

class ProcessData:
    def __init__(self, train_file_url, test_file_url, dev_file_url = None):
        self.train_file_url = train_file_url
        self.test_file_url = test_file_url
        self.dev_file_url = dev_file_url
        # Oracle
        self.oracle = Oracle()
        # Train, test and dev data is a dictionary with the sentences, words and upos
        self.train_data = {}
        self.test_data = {}
        self.dev_data = {}
        self.upos_dict = upos_dictionary
        self.deprel_dict = deprel_dictionary
        self.arcs_dict = arcs_dictionary

    # Dos parametros: url del archivo y el tipo de archivo, limitado a train, test o dev
    def read_conllu_file(self, type_file):
        if type_file == "train":
            url_file = self.train_file_url
        elif type_file == "test":
            url_file = self.test_file_url
        elif type_file == "dev":
            url_file = self.dev_file_url
        else:
            raise Exception("The type of file is not correct")

        # Download the file
        path_to_file = tf.keras.utils.get_file(origin=url_file)

        with open(path_to_file, "r") as f:
            data = f.readlines()

        # Create a dataframe for each sentence
        sentences_dataframes = self.preprocess_data(data)

        # Sentences in format list of lists
        # [['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed'], ['...', '...']]
        data_sentences = self.obtain_data_sentences(sentences_dataframes)

        # [[Al - Zaman : American forces killed Shaikh Abdullah al - Ani , the preacher at the mosque in the town of Qaim , near the Syrian border .], [...]]
        array_sentences = self.create_sentences(data_sentences)

        # All words in a unqiue list
        data_words = self.obtain_data_words(data_sentences)

        # Tags for each word in a list of lists as format of data_sentences
        data_upos = self.obtain_data_upos(sentences_dataframes)

        # Dictionary with the data
        data = {"sentences": array_sentences, "words": data_words, "upos": data_upos, "dataframes": sentences_dataframes}

        # Save the data
        if type_file == "train":
            self.train_data = data
        elif type_file == "test":
            self.test_data = data
        elif type_file == "dev":
            self.dev_data = data


    #########################################################################################
    ####################### Functions to prepare dataframes #################################
    #########################################################################################

    # Separate rows into groups of rows
    # When appears a empty row is the end of a sentence
    # If a row starts with # it doesn't belong to the sentence
    def separate_sentences(self, data):
        sentences, sentence = [], []
        for row in data:
            if row == "\n":
                sentences.append(sentence)
                sentence = []
            elif not row.startswith("#"):
                sentence.append(row.split("\t"))
        return sentences


    # Delete multiword tokens
    # If a word has a "-" in the first column is a multiword token
    def delete_multiword(self, sentence):
        return [row for row in sentence if "-" not in row[0]]
    
    # Function to add Root to each sentence
    # Is added in the first position
    def add_root(self, sentence):
        sentence.insert(0, ["0", "Root", "Root", "_", "_", "_", "_", "_", "_", "_"])
        return sentence
    
    # Function to obtain the tree of each sentence
    def obtain_arcs(self, sentence):
        # Make an array of the column 6 (HEAD) of the sentence
        tree = []
        for row in sentence:
            arc = (int(row[0]), int(row[6]))
            tree.append(arc)

        return tree
    
    # Function to determine if a sentence is projective or not
    # A sentence is projective if the arcs don't cross
    def sentence_is_projective(self, arcs):
        for (i,j) in arcs:
            for (k,l) in arcs:
                if (i,j) != (k,l) and min(i,j) < min(k,l) < max(i,j) < max(k,l):
                    return False
        return True


    # Convert each sentence into a dataframe with a word per row
    # Each column are: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC
    # Delete multiword tokens
    # Change UPOS for a number
    # Max length of a sentence is 128
    def preprocess_data(self, data):
        # Separate data into sentences
        sentences = self.separate_sentences(data)
        dataframes = []

        for sentence in sentences:
            sentence_without_multiword = self.delete_multiword(sentence)
            arcs = self.obtain_arcs(sentence_without_multiword)
            if self.sentence_is_projective(arcs):
                sentence_with_root = self.add_root(sentence_without_multiword)
                dataframes.append(
                    pd.DataFrame(
                        sentence_with_root,
                        columns=[
                            "ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "DEPS", "MISC",
                        ],
                    )
                )
        return dataframes

    #########################################################################################
    #################### Functions to obtain sentences and words ############################
    #########################################################################################

    # Function for obtain the words of each sentence
    def obtain_data_sentences(self, sentences):
        return [sentence["FORM"].tolist() for sentence in sentences]


    # Function for obtain all the words of the dataset
    def obtain_data_words(self, sentences_words):
        return list(chain.from_iterable(sentences_words))


    # Function that creates an array with the words of each sentence separated by spaces
    def create_sentences(self, data_sentences):
        sentences = []
        for sentence in data_sentences:
            if type(sentence[0]) is str:
                sentences.append(" ".join(sentence))
            elif type(sentence[0]) is np.int32:
                sentences.append(" ".join(str(i) for i in sentence))
        return sentences

    #########################################################################################
    #################### Functions to obtain UPOS tags ######################################
    #########################################################################################

    # Function for obtain all the upos of the dataset
    def obtain_data_upos(self, sentences):
        return [sentence["UPOS"].tolist() for sentence in sentences]
    
    #################################################################################################
    #################### Functions to obtain the data for the model #################################
    #################################################################################################

    # Function to prepare samples for the model
    def create_samples(self, type_data, new_samples = False):
        if type_data == "train":
            data = self.train_data
        elif type_data == "test":
            data = self.test_data
        elif type_data == "dev":
            data = self.dev_data
        else:
            raise Exception("The type of data is not correct")
        
        # If folder samples/ doesn't exist, create it
        if not os.path.exists("samples/"):
            os.makedirs("samples/")
            new_samples = True

        # If the samples don't exist, create them
        if self.samples_exist(type_data) and not new_samples:
            samples = self.load_samples(type_data)
        else:
            print("Creating samples...")
            samples = []
            for dataframe in data["dataframes"]:
                samples.append(self.oracle.start_oracle(dataframe))
            self.save_samples(samples, type_data)

        # Save the samples in the dataset
        data["samples"] = samples

    # Function to obtain the samples of a sentence
    def obtain_tree(self, sentence, print_tree = False):
        steps = self.oracle.start_oracle(sentence)
        if print_tree:
            for step in steps:
                print(step)
        return steps
    
    # Function to save the samples in a file
    def save_samples(self, data, type_data):
        with open("samples/" + type_data + "_samples.json", "w") as f:
            json.dump(data, f)

    # Function to load the samples from a file
    def load_samples(self, type_data):
        with open("samples/" + type_data + "_samples.json", "r") as f:
            data = json.load(f)
        return data

    # Function to revise if the samples exist
    def samples_exist(self, type_data):
        return os.path.exists("samples/" + type_data + "_samples.json")
    
