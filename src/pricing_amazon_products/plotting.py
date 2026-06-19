import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_feature_distribution(df: pd.DataFrame, col: str, title: str, symb: str):
    '''
    Plot the distribution of a numeric variable using a boxplot and histogram.

    Parameters
    ----------
    - df (pd.DataFrame): Dataframe containing the variable to plot.
    - col (str): Name of the numeric column to visualize.
    - title (str): Plot title.
    - symb (str): Unit symbol to append to the mean and median labels.
    '''
    _, ax = plt.subplots(2, 1, sharex=True, figsize=(12,5),gridspec_kw={"height_ratios": (.2, .8)})
    ax[0].set_title(title,fontsize=18)
    sns.boxplot(x=col, data=df, ax=ax[0])
    ax[0].set(yticks=[])
    sns.distplot(df[col], ax=ax[1])
    ax[1].set_xlabel(col, fontsize=16)
    plt.axvline(df[col].mean(), color='darkgreen', linewidth=2.2, label='mean=' + str(np.round(df[col].mean(),2)) + symb)
    plt.axvline(df[col].median(), color='red', linewidth=2.2, label='median='+ str(np.round(df[col].median(),2)) + symb)
    plt.legend(bbox_to_anchor=(1, 1.03), ncol=1, fontsize=17, fancybox=True, shadow=True, frameon=True)
    plt.tight_layout()
    plt.show()