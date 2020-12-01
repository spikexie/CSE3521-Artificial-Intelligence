from collections import defaultdict
import numpy as np


def file_reader(file_path, label):
    list_of_lines = []
    list_of_labels = []

    for line in open(file_path):
        line = line.strip()
        if line == "":
            continue
        list_of_lines.append(line)
        list_of_labels.append(label)

    return (list_of_lines, list_of_labels)


def data_reader(source_directory):
    positive_file = source_directory + "Positive.txt"
    (positive_list_of_lines, positive_list_of_labels) = file_reader(file_path=positive_file, label=1)

    negative_file = source_directory + "Negative.txt"
    (negative_list_of_lines, negative_list_of_labels) = file_reader(file_path=negative_file, label=-1)

    neutral_file = source_directory + "Neutral.txt"
    (neutral_list_of_lines, neutral_list_of_labels) = file_reader(file_path=neutral_file, label=0)

    list_of_all_lines = positive_list_of_lines + negative_list_of_lines + neutral_list_of_lines
    list_of_all_labels = np.array(positive_list_of_labels + negative_list_of_labels + neutral_list_of_labels)

    return list_of_all_lines, list_of_all_labels


def evaluate_predictions(test_set, test_labels, trained_classifier):
    correct_predictions = 0
    predictions_list = []
    prediction = -1
    for dataset, label in zip(test_set, test_labels):
        probabilities = trained_classifier.predict(dataset)
        if probabilities[0] >= probabilities[1] and probabilities[0] >= probabilities[-1]:
            prediction = 0
        elif probabilities[1] >= probabilities[0] and probabilities[1] >= probabilities[-1]:
            prediction = 1
        else:
            prediction = -1
        if prediction == label:
            correct_predictions += 1
            predictions_list.append("+")
        else:
            predictions_list.append("-")

    print("Total Sentences correctly: ", len(test_labels))
    print("Predicted correctly: ", correct_predictions)
    print("Accuracy: {}%".format(round(correct_predictions / len(test_labels) * 100, 5)))

    return predictions_list, round(correct_predictions / len(test_labels) * 100)


class NaiveBayesClassifier(object):
    def __init__(self, n_gram=1, printing=False):
        self.prior = []
        self.conditional = []
        self.V = []
        self.n = n_gram

    def word_tokenization_dataset(self, training_sentences):
        training_set = list()
        for sentence in training_sentences:
            cur_sentence = list()
            for word in sentence.split(" "):
                cur_sentence.append(word.lower())
            training_set.append(cur_sentence)
        return training_set

    def word_tokenization_sentence(self, test_sentence):
        cur_sentence = list()
        for word in test_sentence.split(" "):
            cur_sentence.append(word.lower())
        return cur_sentence

    def compute_vocabulary(self, training_set):
        vocabulary = set()
        for sentence in training_set:
            for word in sentence:
                vocabulary.add(word)
        V_dictionary = dict()
        dict_count = 0
        for word in vocabulary:
            V_dictionary[word] = int(dict_count)
            dict_count += 1
        return V_dictionary

    def train(self, training_sentences, training_labels):
        # Get number of sentences in the training set
        N_sentences = len(training_sentences)

        # This will turn the training_sentences into the format described in the PPT
        training_set = self.word_tokenization_dataset(training_sentences)

        # Get vocabulary (dictionary) used in training set
        self.V = self.compute_vocabulary(training_set)

        # Get set of all classes
        all_classes = set(training_labels)

        # -----------------------#
        # -------- TO DO --------#
        # -------- TO DO --------#
        # Note that, you have to further change each sentence in training_set into a binary BOW representation, given self.V

        max = np.zeros(shape=(N_sentences, len(self.V)))
        Vlist = list(self.V.keys())
        for i in range(0, N_sentences):
            for word in training_set[i]:
                if word in Vlist:
                    index = Vlist.index(word)
                    max[i][index] = 1
                    #print(max[i])
        #print(max)


        # Compute the conditional probabilities and priors from training data, and save them in:
        # self.prior
        # self.conditional
        # You can use any data structure you want.
        # You don't have to return anything. self.conditional and self.prior will be called in def predict():


        positive = 0.0
        neutral = 0.0
        negative = 0.0
        for ele in training_labels:
            if ele == -1:
                negative += 1.0
            if ele == 0:
                neutral += 1.0
            if ele == 1:
                positive += 1.0
        self.prior.append(positive / (len(training_labels)))
        self.prior.append(neutral / (len(training_labels)))
        self.prior.append(negative / (len(training_labels)))

        self.conditional = np.zeros(shape=(6, len(Vlist)))
        for i in range(0, max.shape[1]):
            posi = 0.0
            Nposi = 0.0
            neu = 0.0
            Nneu = 0.0
            nega = 0.0
            Nnega = 0.0
            for j in range(0, max.shape[0]):
                if max[j][i] == 1 and training_labels[j] == 1:
                    posi += 1.0
                if max[j][i] == 0 and training_labels[j] == 1:
                    Nposi += 1.0
                if max[j][i] == 1 and training_labels[j] == 0:
                    neu += 1.0
                if max[j][i] == 0 and training_labels[j] == 0:
                    Nneu += 1.0
                if max[j][i] == 1 and training_labels[j] == -1:
                    nega += 1.0
                if max[j][i] == 0 and training_labels[j] == -1:
                    Nnega += 1.0
            self.conditional[0][i] = (posi+1) / (positive+1)
            self.conditional[3][i] = (Nposi+1) / (positive+1)
            self.conditional[1][i] = (neu+1) / (neutral+1)
            self.conditional[4][i] = (Nneu+1) / (neutral+1)
            self.conditional[2][i] = (nega+1) / (negative+1)
            self.conditional[5][i] = (Nnega+1) / (negative+1)


        # Compute the conditional probabilities and priors from training data, and save them in:
        # self.prior
        # self.conditional
        # You can use any data structure you want.
        # You don't have to return anything. self.conditional and self.prior will be called in def predict():

    def predict(self, test_sentence):
        label_probability = {
            0: 0,
            1: 0,
            -1: 0,
        }

        # This will tokenize the test_sentence. test_sentence[n] will be the "n-th" word in a sentence (n starts from 0)
        test_sentence = self.word_tokenization_sentence(test_sentence)

        # -----------------------#
        # -------- TO DO --------#
        # -------- TO DO --------#
        # Based on test_sentence, please first turn it into the binary BOW representation (given self.V) and compute the log probability
        #self.prior = np.log(self.prior)
        #self.conditional = np.log(self.conditional)
        max = np.zeros(len(self.V))
        Vlist = list(self.V.keys())
        for word in test_sentence:
            if word in Vlist:
                index = Vlist.index(word)
                max[index] = 1

        label_probability[0] += np.log(self.prior[1])
        label_probability[1] += np.log(self.prior[0])
        label_probability[-1] += np.log(self.prior[2])
        for i in range(0, len(max)):
            if max[i] == 1:
                label_probability[1] += np.log(self.conditional[0][i])
                label_probability[0] += np.log(self.conditional[1][i])
                label_probability[-1] += np.log(self.conditional[2][i])
            else:
                label_probability[1] += np.log(self.conditional[3][i])
                label_probability[0] += np.log(self.conditional[4][i])
                label_probability[-1] += np.log(self.conditional[5][i])




        # Return a dictionary of log probability for each class for a given test sentence:
        # i,e, {0: -39.39854137691295, 1: -41.07638511893377, -1: -42.93948478571315}
        # Please follow the PPT to first perform log (you may use np.log) to each probability terms and sum them.

        return label_probability


if __name__ == '__main__':
    train_folder = "data-sentiment/train/"
    test_folder = "data-sentiment/test/"

    training_sentences, training_labels = data_reader(train_folder)
    test_sentences, test_labels = data_reader(test_folder)

    NBclassifier = NaiveBayesClassifier(n_gram=1)
    NBclassifier.train(training_sentences, training_labels)

    results, acc = evaluate_predictions(test_sentences, test_labels, NBclassifier)

    # Please list your collaborators here:
    # E.g., Wei-Lun Chao, chao.209
    # ...
