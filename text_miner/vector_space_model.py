import numpy as np


class VectorSpaceModel:
    def __init__(self, lang, matrix=None, samples_id=None, vocabulary=None):
        self.language = lang
        self.matrix = matrix or np.array([[]])
        self.classes = []
        self.vocabulary = vocabulary or []
        self.samples_id = samples_id or []

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
            raise ValueError(f"Feature '{feat}' does not exist in vocabulary")

        self.matrix = np.delete(self.matrix, index, axis=1)
        self.vocabulary.remove(feat)

    def add_sample(self, sample_id, sample_vector, class_name):
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
            raise ValueError(f"Vocabulary has repeated terms")
        return self.vocabulary

    def is_valid_vocabulary(self):
        vocab_set = set(self.vocabulary)
        return len(vocab_set) == len(self.vocabulary)

    def get_model(self):
        """
        Retorna el modelo contruido en una tupla (X,y). Donde la X es una matriz de distribucion
        de frecuencias por documentos, columnas por filas respectiuvamente; la y es un vector que
        contiene el valor cofificado de la clase asociada a cada documento respectivamente.
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

    # def get_csr_matrix(self):
    #     from scipy.sparse import csr_matrix
    #     return csr_matrix(self.matrix)


if __name__ == '__main__':
    v = VectorSpaceModel('spanish')
    sample1 = {'hoy': 12, 'voy': 5, 'a': 41, 'jugar': 11}
    sample2 = {'hoy': 1, 'pierdo': 3, 'la': 124, 'jugada': 88, 'increible': 1}
    sample3 = {'hoy': 7, 'besare': 3, 'a': 7, 'la': 3, 'increible': 12, 'mujer': 5}
    v.add_sample('ejemplo', sample1, '1')
    v.add_sample('ejemplo2', sample2, '2')
    v.add_sample('ejemplo3', sample3, '3')
    v.remove_feature('jugada')
    print("Number of words:", v.count_features())
    print("Number of documents:", v.count_samples())
