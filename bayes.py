import sys
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB

def read_data():
    stoplist_file = open('data/stoplist.txt')
    traindata_file = open('data/traindata.txt')
    trainlabel_file = open('data/trainlabels.txt')
    testdata_file = open('data/testdata.txt')
    testlabel_file = open('data/testlabels.txt')
    
    stop_words = []
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    
    for line in stoplist_file:
        line = line.replace('\n', '')
        stop_words.append(line)
        
    for line in traindata_file:
        line = line.replace('\n', '')
        train_data.append(line)
        
    for line in trainlabel_file:
        line = line.replace('\n', '')
        train_label.append(int(line))
    
    for line in testdata_file:
        line = line.replace('\n', '')
        test_data.append(line)
    
    for line in testlabel_file:
        line = line.replace('\n', '')
        test_label.append(int(line))
        
    return (stop_words, train_data, train_label, test_data, test_label)

def preprocess(stop_words, train_data):
    
    vocabulary = []
    
    for line in train_data:
        line = line.replace('\n', '')
        line = line.split(' ')
        for word in line:
            if word not in stop_words and word not in vocabulary and len(word) > 0:
                vocabulary.append(word)
    vocabulary.sort()
    return vocabulary

def convert_to_feature(vocabulary, train_data, train_label, test_data, test_label):
    
    train_x = np.zeros((len(train_data), len(vocabulary)))
    test_x = np.zeros((len(test_data), len(vocabulary)))
    
    train_count = 0
    
    for line in train_data:
        line = line.replace('\n', '')
        line = line.split(' ')
        for word in line:
            if word in vocabulary:
                index = vocabulary.index(word)
                train_x[train_count][index] = 1
                
        train_count += 1
        
    test_count = 0
    
    for line in test_data:
        line = line.replace('\n', '')
        line = line.split(' ')
        for word in line:
            if word in vocabulary:
                index = vocabulary.index(word)
                test_x[test_count][index] = 1
                
        test_count += 1
    
    train_y = list(map(int, train_label))
    test_y = list(map(int, test_label))
    
    train_x = pd.DataFrame(train_x, columns = vocabulary)
    test_x = pd.DataFrame(test_x, columns = vocabulary)
    
    train_y = pd.DataFrame(train_y, columns = ['label'])
    test_y =  pd.DataFrame(test_y, columns = ['label'])
    
    return (train_x, train_y, test_x, test_y)

def create_output(train_accuracy, test_accuracy, train_accuracy_sk, test_accuracy_sk):
    
    file = open('output.txt', 'w+')
    
    file.write('##### Naive Bayes Implementation Output #####\n\n')
    file.write('Train accuracy: ' + str(train_accuracy) + ' %\n')
    file.write('Test accuracy: ' + str(test_accuracy) + ' %\n')
    file.write('\n\n##### Naive Bayes SKLearn Output #####\n\n')
    file.write('Train accuracy: ' + str(train_accuracy_sk) + ' %\n')
    file.write('Test accuracy: ' + str(test_accuracy_sk) + ' %\n')
    file.close()

class NaiveBayes():
    
    def __init__(self, vocabulary):
        self.attribute_estimates = {}
        self.class_estimates = {}
        self.vocabulary = vocabulary
        
    def fit(self, x, y):
        c = 0
        attributes = x.columns.values
        labels = y.columns.values
        
        for label in labels:
            value, count = np.unique(y[label], return_counts = True)
            value_count = dict(zip(value, count))
            for key, value in value_count.items():
                self.class_estimates[key] = value_count[key] / y.shape[0]
         
        for attribute in attributes:
            value, count = np.unique(x[attribute], return_counts = True)
            
            word_dict = {}
            
            for v in value:
                y_l = y.values
                index = np.where(x[attribute] == v)
                corr_y = np.take(y_l, index)[0]
                ops, total = np.unique(corr_y, return_counts = True)
                ops_total = dict(zip(ops, total))
                total = np.sum(list(ops_total.values()))
                
                for key, val in ops_total.items():
                    ops_total[key] = (ops_total[key] + 1) / (total + 2)
                
                # Handle values which doesn't appear in dictionary
                if(len(ops_total) == 1):
                    key = list(ops_total.keys())
                    if(key[0] == 0):
                        ops_total[1] = 1 - ops_total[key[0]]
                    else:
                        ops_total[0] = 1 - ops_total[key[0]]
                
                word_dict[v] = ops_total
                
            self.attribute_estimates[attribute] = word_dict
            
           
    def score(self, x, y):
        pred_future = 1
        pred_saying = 1
        y_hat = []
        for i in range(len(x)):
            j = 0
            attributes = x[i].split(' ')
            for attribute in attributes:
                if attribute not in self.vocabulary:
                    continue
                else:
                    pred_future *= self.attribute_estimates[attribute][1.0][1]
                    pred_saying *= self.attribute_estimates[attribute][1.0][0]
                    j += 1
            pred_future *= self.class_estimates[1]
            pred_saying *= self.class_estimates[0]
            if(pred_future > pred_saying):
                y_hat.append(1)
            else:
                y_hat.append(0)
            pred_future = 1
            pred_saying = 1
        accuracy = self.calculate_accuracy(y_hat, y)
        return accuracy
        
    def calculate_accuracy(self, y_hat, y):
        y_hat = np.asarray(y_hat)
        y = np.asarray(y)
        
        count = np.equal(y_hat, y)
        value, count = np.unique(count, return_counts = True)
        val_count = dict(zip(value, count))
        
        accuracy = 1 - (val_count[False] / y_hat.shape[0])
        
        return accuracy

def main():
    
    # Read the data from text file
    (stop_words, train_data, train_label, test_data, test_label) = read_data()
    
    # Create a vocabulary of words 
    vocabulary = preprocess(stop_words, train_data)
    
    # Convert the data into feature vector
    (train_x, train_y, test_x, test_y) = convert_to_feature(vocabulary, train_data, train_label, test_data, test_label)
    
    # Instantiate Naive Bayes classifier object
    nb = NaiveBayes(vocabulary)
    
    # Fit model on training data
    nb.fit(train_x, train_y)
    
    # Accuracy for train data
    train_accuracy = round(nb.score(train_data, train_label) * 100, 2)
    
    # Accuracy for test data
    test_accuracy = round(nb.score(test_data, test_label) * 100, 2)
    
    # Use SKlearn to verify
    clf = MultinomialNB()
    
    # Fit train data to our model
    clf.fit(train_x, train_y)
    
    # SKlearn accuracy for train data
    train_accuracy_sk = round(clf.score(train_x, train_y) * 100, 2)
    
    # SKlearn accuracy for test data
    test_accuracy_sk = round(clf.score(test_x, test_y) * 100, 2)
    
    # Write output to file
    create_output(train_accuracy, test_accuracy, train_accuracy_sk, test_accuracy_sk)

if __name__ == '__main__':
    main()

