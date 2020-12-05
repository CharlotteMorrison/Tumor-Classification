import pandas as pd
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
import seaborn as sn


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
        plt.close()

    def split_test_train(self, test_ratio=0.2):
        X = self.df.copy(deep=True)
        X = X.drop(X.columns[[0]], axis=1)
        # TODO create a pipeline

        X = SimpleImputer(strategy='median').fit_transform(X)
        X = preprocessing.scale(X)
        # X = StandardScaler().fit_transform(X)  # produces negative values, cant use for QDA
        # X = QuantileTransformer().fit_transform(X)
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=self.labels[1:])
        y = self.df[['diagnosis']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42)
        return X_train, X_test, y_train, y_test

    def create_scatter_matrix(self, df):
        # save the complete correlation matrix to a file
        correlation = df.corr()
        correlation.to_csv('outputs/correlations/all_correlations.csv')
        sn.heatmap(correlation, annot=False)
        plt.savefig('outputs/correlations/all_correlations.png')
        # plt.show()
        plt.close()

        # create matrix of each 3 measures for attributes
        attributes = df.columns.tolist()
        for i in range(10):
            matrix_att = [attributes[i], attributes[i + 10], attributes[i + 20]]
            matrix_df = df[matrix_att]
            corr_matrix = matrix_df.corr()
            sn.heatmap(corr_matrix, annot=True)
            name = attributes[i].split('_')[0]
            plt.savefig('outputs/correlations/{}_correlation_matrix.png'.format(name))
            plt.close()
        # create a matrix for each grouping of attributes (all mean, se, worst)
        matrix_df = [df[attributes[:10]], df[attributes[10:20]], df[attributes[20:30]]]
        labels = ['mean', 'se', 'worst']
        i = 0
        for matrix in matrix_df:
            correlation = matrix.corr()
            correlation.to_csv('outputs/correlations/{}_correlations.csv'.format(labels[i]))
            sn.heatmap(correlation, annot=False)
            plt.tight_layout()
            plt.savefig('outputs/correlations/{}_correlations.png'.format(labels[i]))
            # plt.show()
            i += 1
            plt.close()
