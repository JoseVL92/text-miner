import os
import pickle

import numpy as np


class VectorSpaceModel:
    def __init__(self, *, matrix=None, samples_id=None, vocabulary=None, classes=None):
        self.matrix = matrix or np.array([[]])
        self.classes = classes or []
        self.vocabulary = vocabulary or []
        self.samples_id = samples_id or []

        if vocabulary:
            self.get_vocabulary()
        if samples_id and classes:
            if len(samples_id) != len(classes):
                raise ValueError("Samples list and classes list must have the same size")
        if matrix and samples_id:
            self.count_samples()
        if matrix and vocabulary:
            self.count_features()

    def add_features(self, feats):
        for feat in feats:
            if feat not in self.vocabulary:
                # vocab_index = len(self.vocabulary)
                self.vocabulary.append(feat)
                if self.matrix.size > 0:
                    to_add = np.zeros((len(self.samples_id), 1))
                    self.matrix = np.append(self.matrix, to_add, axis=1)

    def remove_feature(self, feat):
        try:
            index = self.vocabulary.index(feat)
        except ValueError:
            raise ValueError("Feature '{}' does not exist in vocabulary".format(feat))

        self.matrix = np.delete(self.matrix, index, axis=1)
        self.vocabulary.remove(feat)

    def add_sample(self, sample_id, sample_vector, class_name):
        # if sample_id in self.samples_id:
        #     raise ValueError("Vector with ID {} is already present in this VSM")
        self.add_features(list(sample_vector.keys()))
        sample_as_array = np.zeros((1, len(self.vocabulary)))

        for word, count in sample_vector.items():
            word_index = self.vocabulary.index(word)
            sample_as_array[0, word_index] = count

        self.samples_id.append(sample_id)
        self.classes.append(class_name)

        if self.matrix.size == 0:
            self.matrix = sample_as_array
        else:
            self.matrix = np.append(self.matrix, sample_as_array, axis=0)

    def remove_sample(self, sample_id):
        index = self.samples_id.index(sample_id)
        self.matrix = np.delete(self.matrix, index, axis=0)
        self.samples_id.remove(sample_id)
        self.classes.pop(index)

    def count_features(self):
        assert len(self.vocabulary) == self.matrix.shape[
            1], "Length of matrix correspondent dimension and vocabulary list are different"
        return len(self.vocabulary)

    def count_samples(self):
        assert len(self.samples_id) == self.matrix.shape[
            0], "Length of matrix correspondent dimension and samples_id list are different"
        return len(self.samples_id)

    def get_vocabulary(self):
        if not self.is_valid_vocabulary():
            raise ValueError("Vocabulary has repeated terms")
        return self.vocabulary

    def is_valid_vocabulary(self):
        vocab_set = set(self.vocabulary)
        return len(vocab_set) == len(self.vocabulary)

    def get_model(self):
        """
        Retorna el modelo construido en una tupla (X,y). Donde la X es una matriz de distribucion
        de frecuencias por documentos, columnas por filas respectivamente; la y es un vector que
        contiene el valor codificado de la clase asociada a cada documento respectivamente.
        """
        return self.matrix, np.array(self.classes)

    def fit_sample(self, sample_vector):
        # np_sample = np.zeros((1, self.matrix.shape[1]))
        np_sample = np.zeros((1, len(self.vocabulary)))

        for word, count in sample_vector.items():
            try:
                word_index = self.vocabulary.index(word)
                np_sample[0, word_index] = count
            except ValueError:
                continue

        return np_sample

    def save_matrix(self, file_path):
        try:
            with open(file_path, 'wb') as matrix_file:
                np.save(matrix_file, self.matrix)
        except:
            np.save(file_path, self.matrix)

    def load_matrix(self, file_path):
        self.matrix = np.load(file_path)

    def save_classes(self, file_path):
        with open(file_path, 'wb') as classes_file:
            pickle.dump(self.classes, classes_file)

    def load_classes(self, file_path):
        with open(file_path, 'rb') as f:
            self.classes = pickle.load(f)

    def save_samples_id(self, file_path):
        with open(file_path, 'wb') as samples_file:
            pickle.dump(self.samples_id, samples_file)

    def load_samples_id(self, file_path):
        with open(file_path, 'rb') as f:
            self.samples_id = pickle.load(f)

    def save_vocabulary(self, file_path):
        with open(file_path, 'wb') as vocab_file:
            pickle.dump(self.vocabulary, vocab_file)

    def load_vocabulary(self, file_path):
        with open(file_path, 'rb') as f:
            self.vocabulary = pickle.load(f)

    def save_vsm(self, dir_path, matrix_name='matrix.vsm.npy', vocabulary_name='vocabulary.vsm',
                 classes_name='classes.vsm', samples_id_name='samples.vsm'):
        if not os.path.isdir(dir_path):
            raise ValueError("{} does not exist or is not a directory".format(dir_path))
        self.save_matrix(os.path.join(dir_path, matrix_name))
        self.save_classes(os.path.join(dir_path, classes_name))
        self.save_samples_id(os.path.join(dir_path, samples_id_name))
        self.save_vocabulary(os.path.join(dir_path, vocabulary_name))

    def load_vsm(self, dir_path, matrix_name='matrix.vsm.npy', vocabulary_name='vocabulary.vsm',
                 classes_name='classes.vsm', samples_id_name='samples.vsm'):
        if not os.path.isdir(dir_path):
            raise ValueError("{} does not exist or is not a directory".format(dir_path))
        matrix_path = os.path.join(dir_path, matrix_name)
        classes_path = os.path.join(dir_path, classes_name)
        samples_path = os.path.join(dir_path, samples_id_name)
        vocab_path = os.path.join(dir_path, vocabulary_name)
        paths = (matrix_path, classes_path, samples_path, vocab_path)
        for p in paths:
            if not os.path.isfile(p):
                raise OSError("{} is not a file".format(p))
        self.load_matrix(matrix_path)
        self.load_classes(classes_path)
        self.load_samples_id(samples_path)
        self.load_vocabulary(vocab_path)

# def get_csr_matrix(self):
#     from scipy.sparse import csr_matrix
#     return csr_matrix(self.matrix)
