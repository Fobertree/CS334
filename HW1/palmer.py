"""
Pandas DataFrame Manipulation with Palmer Penguins Dataset
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import pandas as pd
import numpy as np

def load_csv(inputfile : str) -> pd.DataFrame:
    """
    Load the csv as a pandas data frame
    
    Parameters
    ----------
    inputfile : string
        filename of the csv to load

    Returns
    -------
    csvdf : pandas.DataFrame
        return the pandas dataframe with the contents
        from the csv inputfile
    """

    csvdf = pd.read_csv(inputfile)
    return csvdf


def remove_na(inputdf : pd.DataFrame, colname : str) -> pd.DataFrame:
    """
    Remove the rows in the dataframe with NA as values 
    in the column specified.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to check and remove rows with NA

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """

    # print(len(inputdf[inputdf[colname].notna()]), len(inputdf[colname]))
    outputdf = inputdf[inputdf[colname].notna()]
    
    return outputdf


def onehot(inputdf : pd.DataFrame, colname : str) -> pd.DataFrame:
    """
    Convert the column in the dataframe into a one hot encoding.
    The newly converted columns should be at the end of the data
    frame and you should also drop the original column.
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe
    colname : string
        Name of the column to one-hot encode

    Returns
    -------
    outputdf : pandas.DataFrame
        return the pandas dataframe with the modified contents
    """

    one_hot = pd.get_dummies(inputdf[colname])
    inputdf.drop(columns=colname, inplace=True)
    outputdf = inputdf.join(one_hot)
    return outputdf


def to_numeric(inputdf: pd.DataFrame) -> np.ndarray:
    """
    Extract all the 
    
    Parameters
    ----------
    inputdf : pandas.DataFrame
        Input dataframe

    Returns
    -------
    outputnp : numpy.ndarray
        return the numeric contents of the input dataframe as a 
        numpy array
    """
    outputnp = inputdf.to_numpy()
    return outputnp


def main():
    # Load data
    df = load_csv("data/penguins.csv")

    # Remove NA
    df = remove_na(df, "species")

    # One hot encoding
    df = onehot(df, "species")

    # Convert to numeric
    df_np = to_numeric(df)

    print(df)
    print(df_np)

if __name__ == "__main__":
    main()
