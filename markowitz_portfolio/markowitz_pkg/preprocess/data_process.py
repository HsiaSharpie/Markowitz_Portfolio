import sys
sys.path.append("../../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import Namespace

# args = Namespace(
#     file = '../data/stock_price.xlsx',
#     portion = 0.8
# )

class Dataset(object):
    def __init__(self, dataset, portion):
        price_df = self.read_file(dataset, portion)
        self.price_df = price_df

        # Convert the data from price to return
        self.return_df = self.price_df.pct_change()

        # Calculate other statistics
        self.mean_returns = self.return_df.mean()
        self.cov_matrix = self.return_df.cov()


    def read_file(self, dataset, portion):
        split_filename = dataset.split('.')

        if 'xlsx' in split_filename:
            price_df = pd.read_excel(dataset)

        elif 'csv' in split_filename:
            price_df = pd.read_csv(dataset)

        length_time = len(price_df)
        days = int(length_time * portion)

        price_df = price_df.set_index('Date')
        price_df = price_df.iloc[:days, :]
        return price_df

    def plot_price(self):
        plt.figure(figsize=(10, 8))

        # plot the prices changes
        for col in self.price_df.columns.values:
            plt.plot(self.price_df.index, self.price_df[col], alpha=0.5, label=col)

        plt.legend(loc='upper left', fontsize=12)
        plt.ylabel('price')
        plt.show()


    def plot_return(self):
        plt.figure(figsize=(10, 8))

        # plot the prices changes
        for col in self.return_df.columns.values:
            plt.plot(self.return_df.index, self.return_df[col], alpha=0.5, label=col)

        plt.legend(loc='upper left', fontsize=12)
        plt.ylabel('return')
        plt.show()

# if __name__ == '__main__':
#     dataset = Dataset(args.file, args.portion)
