from exploration import DataExplorer
from classifiers import DataClassifier

if __name__ == "__main__":
    # the dataset has 3 values for each of the attributes: mean, standard error and worst (largest)
    cancer_labels = ['diagnosis']
    attributes = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
                  'concavity', 'concave_points', 'symmetry', 'fractal_dimension']

    # there is no list of labels, so I will make one...
    mean_labels = []
    se_labels = []
    worst_labels = []
    for att in attributes:
        mean_labels.append(att + '_mean')
        se_labels.append(att + '_se')
        worst_labels.append(att + '_worst')

    cancer_labels.extend(mean_labels)
    cancer_labels.extend(se_labels)
    cancer_labels.extend(worst_labels)

    cancer_file = "datasets/wdbc.data"

    cancer = DataExplorer(cancer_file, cancer_labels)

    # get basic information about the whole dataset
    cancer.get_data_structure()

    # split the dataset for train/test
    cancer_x_train, cancer_x_test, cancer_y_train, cancer_y_test = cancer.split_test_train()

    # create a correlation matrix scatter plot for feature reduction
    cancer.create_scatter_matrix(cancer_x_train)

    cancer_classifier = DataClassifier(cancer_x_train, cancer_y_train)
    cancer_classifier.run_sgd()



