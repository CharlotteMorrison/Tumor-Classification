from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pickle
import numpy as np


class DataClassifier:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y.values.ravel()
        self.figure = plt.figure(figsize=(27, 9))
        self.files = ['sgd', 'knn', 'svc', 'gpc', 'dtc', 'rfc', 'mlp', 'abc', 'gnb', 'qda']
        for i in range(len(self.files)):
            self.files[i] = 'models/' + self.files[i]

    def run_sgd(self):
        # Stochastic Gradient Descent Classifier
        classifier = SGDClassifier(random_state=42)
        model = cross_val_predict(classifier, self.train_x, self.train_y, cv=3)
        self.evaluation_report(model, self.files[0])
        pickle.dump(model, open(self.files[0] + '.pkl', 'wb'))

    def run_knn(self, k=3):
        # K Nearest Neighbors Classifier.
        classifier = KNeighborsClassifier(k)

    def run_svc(self, kernel="linear", c=0.025):
        # Support Vector Classifier.
        classifier = SVC(kernel=kernel, C=c)

    def run_gpc(self):
        # Gaussian Process Classifier based on Laplace approximation.
        classifier = GaussianProcessClassifier(1.0 * RBF(1.0))

    def run_dtc(self, max_d=5):
        # Decision Tree Classifier.
        classifier = DecisionTreeClassifier(max_depth=max_d)

    def run_rfc(self, depth=5, estimators=10, features=1):
        # Random Forrest Classifier.
        classifier = RandomForestClassifier(max_depth=depth, n_estimators=estimators, max_features=features)

    def run_mlp(self, a=1, iters=1000):
        # Multi-layer Perceptron classifier.
        classifier = MLPClassifier(alpha=a, max_iter=iters)

    def run_abc(self):
        # AdaBoost Classifier (AdaBoost-SAMME).
        classifier = AdaBoostClassifier()

    def run_gnb(self):
        # Gaussian Naive Bayes Classifier.
        classifier = GaussianNB()

    def run_qda(self):
        # Quadratic Discriminant Analysis.
        classifier = QuadraticDiscriminantAnalysis()

    def evaluation_report(self, model, file_name):
        # todo save all in a big list...
        file_name = file_name + '.txt'
        with open(file_name, 'w') as file:
            file.write('Confusion Matrix \n')
            file.write(np.array_str(confusion_matrix(self.train_y, model)))
            file.write('\nPrecision Score: ')
            file.write(str(precision_score(self.train_y, model, pos_label='M').item()))
            file.write('\nRecall Score: ')
            file.write(str(recall_score(self.train_y, model, pos_label='M').item()))
            file.write('\nF1 Score: ')
            file.write(str(f1_score(self.train_y, model, pos_label='M').item()))
