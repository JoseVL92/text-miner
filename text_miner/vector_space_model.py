import os
import pickle

import numpy as np
from scipy import sparse


class VectorSpaceModel:
    def __init__(self, *, matrix=None, samples_id=None, vocabulary=None, classes=None):
        self.matrix = self._set_matrix_to_coo(matrix)
        self.classes = classes or []
        self.vocabulary = vocabulary or []
        self.samples_id = samples_id or []

        if vocabulary:
            vocab_set = set(vocabulary)
            if len(vocab_set) != len(vocabulary):
                raise ValueError("Vocabulary has repeated terms")
        if samples_id and classes:
            if len(samples_id) != len(classes):
                raise ValueError("Samples list and classes list must have the same size")
        if matrix and samples_id:
            self.count_samples()
        if matrix and vocabulary:
            self.count_features()

    @staticmethod
    def _set_matrix_to_coo(matrix):
        if isinstance(matrix, sparse.coo_matrix):
            return matrix
        elif isinstance(matrix, sparse.spmatrix):
            return matrix.tocoo()
        elif isinstance(matrix, np.ndarray):
            return sparse.coo_matrix(matrix)
        elif matrix is None:
            return sparse.coo_matrix(np.array([[]]))
        else:
            return None

    def add_features(self, feats):
        new_feats = set(feats).difference(self.vocabulary)
        self.vocabulary.extend(new_feats)
        if self.matrix.shape != (1, 0):
            self.matrix.resize(self.matrix.shape[0], len(self.vocabulary))
            assert len(self.vocabulary) == self.matrix.shape[
                1], "vocabulary does not match with matrix correspondent dimension"

    def remove_features(self, feats, no_mean_only=False):
        feats_to_remove = set(feats)
        feats_to_remove_indexes = []
        for f in feats_to_remove:
            try:
                feats_to_remove_indexes.append(self.vocabulary.index(f))
            except ValueError:
                continue
        all_vocab_indexes = range(len(self.vocabulary))
        remaining_feats_indexes = set(all_vocab_indexes).difference(feats_to_remove_indexes)
        remaining_feats_indexes = list(remaining_feats_indexes)
        # CSC for column slicing
        as_csc = self.matrix.tocsc()
        if no_mean_only:
            idx = 0
            while idx < len(feats_to_remove_indexes):
                tester = as_csc[:, [feats_to_remove_indexes[idx]]]
                if tester.max() > 0:
                    remaining_feats_indexes.append(feats_to_remove_indexes.pop(idx))
                else:
                    idx += 1
        remaining_feats_indexes = sorted(remaining_feats_indexes)

        new_matrix = as_csc[:, remaining_feats_indexes]
        self.matrix = new_matrix.tocoo()
        for idx2 in feats_to_remove_indexes:
            self.vocabulary.pop(idx2)
        assert len(self.vocabulary) == self.matrix.shape[
            1], "vocabulary does not match with matrix correspondent dimension"

    def add_sample(self, sample_id, sample_vector, class_name):
        if sample_id in self.samples_id:
            raise ValueError("Vector with ID {} is already present in this VSM".format(sample_id))
        self.add_features(list(sample_vector.keys()))

        data = []
        row = [0]*len(sample_vector)
        col = []

        for word, count in sample_vector.items():
            word_index = self.vocabulary.index(word)
            data.append(count)
            col.append(word_index)

        sparsed_sample = sparse.coo_matrix((data, (row, col)), shape=(1, len(self.vocabulary)))
        if self.matrix.shape == (1, 0):
            self.matrix = sparsed_sample
        else:
            self.matrix = sparse.vstack([self.matrix, sparsed_sample])
        self.samples_id.append(sample_id)
        self.classes.append(class_name)

    def remove_samples(self, samples_id_list):
        samples_to_remove = set(samples_id_list)
        samples_to_remove_indexes = []
        for f in samples_to_remove:
            try:
                samples_to_remove_indexes.append(self.samples_id.index(f))
            except ValueError:
                continue
        all_samples_indexes = range(len(self.samples_id))
        remaining_samples_indexes = set(all_samples_indexes).difference(samples_to_remove_indexes)
        remaining_samples_indexes = list(remaining_samples_indexes)

        # CSR for column slicing
        as_csr = self.matrix.tocsr()
        new_matrix = as_csr[remaining_samples_indexes, :]
        self.matrix = new_matrix.tocoo()
        for idx in samples_to_remove_indexes:
            self.samples_id.pop(idx)
            self.classes.pop(idx)

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
        return len(vocab_set) == self.count_features()

    def get_model(self, matrix_output_format='csr'):
        """
        Retorna el modelo construido en una tupla (X,y). Donde la X es una matriz de distribucion
        de frecuencias por documentos, columnas por filas respectivamente; la y es un vector que
        contiene el valor codificado de la clase asociada a cada documento respectivamente.
        """
        if matrix_output_format not in ('csr', 'coo', 'array'):
            raise ValueError("Format not valid: " + str(matrix_output_format))
        if matrix_output_format == 'coo':
            return self.matrix, np.array(self.classes)
        if matrix_output_format == 'csr':
            return self.matrix.tocsr(), np.array(self.classes)
        return self.matrix.toarray(), np.array(self.classes)

    def transform(self, X, output_format='csr', vocabulary=None):
        """
        Transform a list of raw texts to a matrix fitted to this vsm format
        :param X: List of dictionaries with shape {'term': frequency} for every document to analyze
        :param output_format:
        :param vocabulary:
        :return:
        """
        # np_sample = np.zeros((1, self.matrix.shape[1]))
        # np_sample = np.zeros((1, len(self.vocabulary)))
        if vocabulary is None:
            vocabulary = self.get_vocabulary()
        vocabulary = list(set(vocabulary))
        if output_format not in ('csr', 'coo', 'array'):
            raise ValueError("Format not valid: " + str(output_format))
        data = []
        row = []
        col = []

        for index, vector in enumerate(X):
            for word, count in vector.items():
                try:
                    word_index = vocabulary.index(word)
                    data.append(count)
                    col.append(word_index)
                    row.append(index)
                    # np_sample[0, word_index] = count
                except ValueError:
                    continue
        if output_format == 'csr':
            output = sparse.csr_matrix((data, (row, col)), shape=(len(X), len(vocabulary)))
        else:
            output = sparse.coo_matrix((data, (row, col)), shape=(len(X), len(vocabulary)))
            if output_format == 'array':
                output = output.toarray()
        return output

    def save_matrix(self, file_path):
        try:
            with open(file_path, 'wb') as matrix_file:
                sparse.save_npz(matrix_file, self.matrix)
        except:
            sparse.save_npz(file_path, self.matrix)

    def load_matrix(self, file_path):
        self.matrix = sparse.load_npz(file_path)

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

    def save_vsm(self, dir_path, matrix_name='matrix.vsm.npz', vocabulary_name='vocabulary.vsm',
                 classes_name='classes.vsm', samples_id_name='samples.vsm'):
        if not os.path.isdir(dir_path):
            raise ValueError("{} does not exist or is not a directory".format(dir_path))
        self.save_matrix(os.path.join(dir_path, matrix_name))
        self.save_classes(os.path.join(dir_path, classes_name))
        self.save_samples_id(os.path.join(dir_path, samples_id_name))
        self.save_vocabulary(os.path.join(dir_path, vocabulary_name))

    def load_vsm(self, dir_path, matrix_name='matrix.vsm.npz', vocabulary_name='vocabulary.vsm',
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
