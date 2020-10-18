from exploration import DataExplorer

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

    cancer_df = DataExplorer(cancer_file, cancer_labels)

    cancer_df.get_data_structure()
