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
import seaborn as sn
import matplotlib.pyplot as plt
import itertools


class DataClassifier:
    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y.values.ravel()
        self.figure = plt.figure(figsize=(27, 9))
        self.files = ['sgd', 'knn', 'svc', 'gpc', 'dtc', 'rfc', 'mlp', 'abc', 'gnb', 'qda']
        for i in range(len(self.files)):
            self.files[i] = 'models/' + self.files[i]
        self.results = []

        # classifier parameters
        k = 3               # k nearest
        kernel = 'linear'   # svc
        c = 0.025           # svc
        max_d = 5           # decision tree
        depth = 5           # random forest
        estimators = 10     # random forest
        features = 1        # random forest
        a = 1               # MLP
        iters = 1000        # MLP

        self.classifiers = [
            SGDClassifier(random_state=42),
            KNeighborsClassifier(k),
            SVC(kernel=kernel, C=c),
            GaussianProcessClassifier(1.0 * RBF(1.0)),
            DecisionTreeClassifier(max_depth=max_d),
            RandomForestClassifier(max_depth=depth, n_estimators=estimators, max_features=features),
            MLPClassifier(alpha=a, max_iter=iters),
            AdaBoostClassifier(),
            GaussianNB(),
            # QuadraticDiscriminantAnalysis()
        ]

    def run_classifiers(self):
        # Stochastic Gradient Descent Classifier
        for file, classifier in zip(self.files, self.classifiers):
            if file == 'models/qda':
                self.train_x.dropna(inplace=True)
                print(self.train_x.min())
            model = cross_val_predict(classifier, self.train_x, self.train_y, cv=10)
            self.evaluation_report(model, file)
            pickle.dump(model, open(file + '.pkl', 'wb'))
        self.draw_graphs()

    def evaluation_report(self, model, file_name):
        classifier_name = file_name.split('/')[1]
        file_name = file_name + '.txt'
        matrix = confusion_matrix(self.train_y, model)
        precision = precision_score(self.train_y, model, pos_label='M')
        recall = recall_score(self.train_y, model, pos_label='M')
        f1 = f1_score(self.train_y, model, pos_label='M')
        self.results.append([classifier_name, matrix, precision, recall, f1])

        with open(file_name, 'w') as file:
            file.write('Confusion Matrix \n')
            file.write(np.array_str(matrix))
            file.write('\nPrecision Score: ')
            file.write(str(precision.item()))
            file.write('\nRecall Score: ')
            file.write(str(recall.item()))
            file.write('\nF1 Score: ')
            file.write(str(f1.item()))

    def draw_graphs(self):
        for result in self.results:
            self.plot_confusion_matrix(cm=result[1], name=result[0], normalize=False,
                                       target_names=['negative', 'positive'],
                                       title="{} Confusion Matrix".format(result[0].upper()))
        # accuracy, precision, recall comparison

    def plot_confusion_matrix(self, cm, name, target_names, title='Confusion matrix', cmap=None, normalize=True):
        """
        given a sklearn confusion matrix (cm), make a nice plot
        Arguments
        ---------
        cm:           confusion matrix from sklearn.metrics.confusion_matrix
        name:         classifier name
        target_names: given classification classes such as [0, 1, 2]
                      the class names, for example: ['high', 'medium', 'low']
        title:        the text to display at the top of the matrix
        cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                      see http://matplotlib.org/examples/color/colormaps_reference.html
                      plt.get_cmap('jet') or plt.cm.Blues
        normalize:    If False, plot the raw numbers| If True, plot the proportions

        Usage
        -----
        plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                                  # sklearn.metrics.confusion_matrix
                              normalize    = True,                # show proportions
                              target_names = y_labels_vals,       # list of names of the classes
                              title        = best_estimator_name) # title of graph
        Citation
        ---------
        http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

        """

        accuracy = np.trace(cm) / float(np.sum(cm))
        misclass = 1 - accuracy

        if cmap is None:
            cmap = plt.get_cmap('Blues')

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        if target_names is not None:
            tick_marks = np.arange(len(target_names))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        thresh = cm.max() / 1.5 if normalize else cm.max() / 2
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
        plt.savefig('outputs/graphs/{}_confusion_matrix.png'.format(name))
        plt.show()
