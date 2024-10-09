import matplotlib.pyplot as plt
import pickle
import numpy as np
import seaborn as sns
import pandas as pd

def find_sublist_indices(L, subL):
    len_L = len(L)
    len_subL = len(subL)

    # Iterate through L with a sliding window of size len_subL
    for i in range(len_L - len_subL + 1):
        if L[i:i + len_subL] == subL:
            return list(range(i, i + len_subL))
    return []


def save_results(data, filename='experiment_results1.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_results(filename='experiment_results.pkl', threshold=1e-8):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    data[np.abs(data) < threshold] = 0
    #return pickle.load(file)
    return data


def plot_heatmap(data, x_values, y_values, savename='tmp.png', title="IIA Results for 'is taxonomic'"):
    """
    data : numpy array of size (y_values, x_values)

    x_values : strings to label columns (x axis)

    y_values: strings to use to label rows (y axis)
    """
    # Create a heatmap
    plt.figure(figsize=(10, 5))
    sns.heatmap(data, annot=True, fmt=".2f", cmap="viridis", xticklabels=x_values, yticklabels=y_values)
    plt.title(title)
    plt.ylabel("Model Layer Index")
    plt.xlabel("Token Position Index")
    plt.savefig(savename)
    plt.close()


def load_results(filename):

    df = pd.read_csv(filename, header=None, delimiter='\t', names = ['layer', 'pos', 'value'] )

    relpos_order = {'premise_first':0,
                    'premise_last':1,
                    'conclusion_first':2,
                    'conclusion_last':3,
                    'last':4}

    layer_order = {i: ii for ii,i in enumerate(sorted(set(df['layer'].values)) ) }

    results = np.empty((len(layer_order), len(relpos_order) ))

    for index, row in df.iterrows():
        ii = layer_order[row['layer']]
        jj = relpos_order[row['pos']]
        value = row['value']

        results[ii, jj] = value

    relpos_names = list(relpos_order.keys())
    layer_names = [str(i) for i in list(layer_order.keys())]

    return results, relpos_names, layer_names

