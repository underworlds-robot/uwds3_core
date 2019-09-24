import numpy as np


class SemanticFeaturesExtractor(object):
    """ Extract semantic features from detection class by using word embeddings """

    def __init__(self, embeddings_file_path):
        """ Default constructor """
        self.word_to_vector = {}
        self.index_to_vector = {}
        index = 0
        with open(embeddings_file_path, "r") as file:
            for line in file:
                tokens = line.split()
                vector = np.array([float(i) for i in tokens[2:]]).astype(np.float32)
                self.word_to_vector[tokens[0]] = vector
                self.index_to_vector[index] = vector
                index += 1

    def extract(self, class_id):
        """ Returns the word vector of the given class id """
        return self.index_to_vector[class_id]

    def get_word_vector(self, class_name):
        """ Returns the word vector of the given class name """
        return self.word_to_vector[class_name]
