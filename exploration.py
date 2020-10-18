import pandas as pd
import matplotlib.pyplot as plt


class DataExplorer:
    def __init__(self, data_file, label_list):
        self.df = pd.read_csv(data_file, header=None)
        # drop the first column of id numbers
        self.df = self.df.drop(self.df.columns[[0]], axis=1)
        # rename the columns
        self.labels = label_list
        self.df.columns = self.labels

    def get_data_structure(self):
        print("*************************************************************")
        print("Dataset Head")
        print(self.df.head())
        print("*************************************************************")
        print("Description of the data:")
        print(self.df.info())
        print("*************************************************************")
        print("Summary of the Numerical Attributes")
        print(self.df.describe())
        self.df.hist(bins=50, figsize=(20, 15))
        plt.savefig("outputs/histograms.png")
        plt.show()
