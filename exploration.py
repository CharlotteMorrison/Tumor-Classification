import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split


class DataExplorer:
    def __init__(self, data_file, label_list):
        self.df = pd.read_csv(data_file, header=None)
        # drop the first column of id numbers
        self.df = self.df.drop(self.df.columns[[0]], axis=1)
        # rename the columns
        self.labels = label_list
        self.df.columns = self.labels

    def get_data_structure(self):
        # save the head
        self.df.head().to_csv('outputs/data_head.csv')

        # save the info about each attribute
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        s = buffer.getvalue()
        with open('outputs/data_info.txt', 'w', encoding='utf-8') as f:
            f.write(s)

        # save the description to a csv
        self.df.describe().to_csv('outputs/data_describe.csv')

        # output histograms for all attributes
        self.df.hist(bins=50, figsize=(20, 15))
        plt.savefig("outputs/histograms.png")
        # plt.show()

    def split_test_train(self, test_ratio=0.2):
        X = self.df.copy(deep=True)
        X = X.drop(X.columns[[0]], axis=1)
        y = self.df[['diagnosis']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        return X_train, X_test, y_train, y_test


