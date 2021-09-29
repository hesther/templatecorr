import pandas as pd
import numpy as np
import time

def split_data_df(data, val_frac = 0.1, test_frac = 0.1, shuffle=True, seed = 12345):
    """
    Splits a pandas dataframe.
    
    :param data: Pandas dataframe
    :param val_frac: Fraction of validation data
    :param test_frac: Fraction of test data
    :param shuffle: Boolean whether to shuffle data
    :param seed: Seed for random number generator for shuffling
    
    :return: Pandas dataframe with added dataset column
    """
   
    # Define shuffling
    if shuffle:
        if seed is None:
            np.random.seed(int(time.time()))
        else:
            np.random.seed(seed)
            
        def shuffle_func(x):
            np.random.shuffle(x)
    else:
        def shuffle_func(x):
            pass
    
    #get all indeces
    indeces = data.index.tolist()
    N = len(indeces)
    print ("{} reactions available in the dataset".format(N))
    
    shuffle_func(indeces)
    
    train_end = int((1.0-test_frac-val_frac)*N)
    val_end = int((1.0-test_frac)*N)
    
    for i in indeces [:train_end]:
        data.at[i, "dataset"]= "train"
    for i in indeces [train_end:val_end]:
        data.at[i,"dataset"]="val"    
    for i in indeces [val_end:]:
        data.at[i,"dataset"]="test"
    
    print(data["dataset"].value_counts())

    return data
