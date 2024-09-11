"""
Vectorization Comparison for Computing Sum of Squares
~~~~~~
Follow the instructions in the homework to complete the assignment.
"""
import numpy as np
import pandas as pd
import timeit
from collections import defaultdict
import matplotlib.pyplot as plt

from tqdm import tqdm

def gen_random_samples(n: int) -> np.ndarray:
    """
    Generate n random samples using the
    numpy random.randn module.

    Returns
    ----------
    sample : 1d array of size n
        An array of n random samples
    """

    # TODO: Implement this function
    sample = np.random.randn(n)
    return sample


def sum_squares_for(samples: np.ndarray) -> float:
    """
    Compute the sum of squares using a forloop

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """
    # TODO: Implement this function
    ss = 0

    for i in samples:
        ss += i * i
    
    return ss


def sum_squares_np(samples: np.ndarray) -> float:
    """
    Compute the sum of squares using Numpy's dot module

    Parameters
    ----------
    samples : 1d-array with shape n
        An array of numbers.

    Returns
    -------
    ss : float
        The sum of squares of the samples
    """

    ss = np.dot(samples,samples.T)
    return ss


def time_ss(sample_list : list[int]) -> dict:
    """
    Time it takes to compute the sum of squares
    for varying number of samples. The function should
    generate a random sample of length s (where s is an 
    element in sample_list), and then time the same random 
    sample using the for and numpy loops.

    Parameters
    ----------
    samples : list of length n
        A list of integers to .

    Returns
    -------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.
    """

    ss_dict = defaultdict(list)

    for n in tqdm(sample_list):
        rand_arr = gen_random_samples(n) #fails here
        # for loop
        ssfor_st = timeit.default_timer()
        sum_squares_for(samples=rand_arr)
        ssfor_time = round(timeit.default_timer() - ssfor_st, 4)

        # np dot product
        ssnp_st = timeit.default_timer()
        sum_squares_np(samples=rand_arr)
        ssnp_time = round(timeit.default_timer() - ssnp_st, 4)

        # update dictionary
        ss_dict['n'].append(n)
        ss_dict['ssfor'].append(ssfor_time)
        ss_dict['ssnp'].append(ssnp_time)


    # return as dict just to follow instructions
    return dict(ss_dict) 


def timess_to_df(ss_dict : dict) -> pd.DataFrame:
    """
    Time the time it takes to compute the sum of squares
    for varying number of samples.

    Parameters
    ----------
    ss_dict : Python dictionary with 3 keys: n, ssfor, ssnp.
        The value for each key should be a list, where the 
        ordering of the list follows the sample_list order 
        and the timing in seconds associated with that 
        number of samples.

    Returns
    -------
    time_df : Pandas dataframe that has n rows and 3 columns.
        The column names must be n, ssfor, ssnp and follow that order.
        ssfor and ssnp should contain the time in seconds.
    """

    time_df = pd.DataFrame.from_dict(ss_dict)

    return time_df


def main():
    # generate 100 samples
    samples = gen_random_samples(100)
    # call the for version
    ss_for = sum_squares_for(samples)
    # call the numpy version
    ss_np = sum_squares_np(samples)
    # make sure they are approximately the same value
    import numpy.testing as npt
    npt.assert_almost_equal(ss_for, ss_np, decimal=5)

    input_dims = [i for i in range(1,10**7, 10000)]

    df = timess_to_df(time_ss(input_dims))
    #df['index'] = input_dims
    df.set_index('n', inplace=True)
    #df.to_csv("sumsquare.csv")
    ax = plt.plot(df)
    ax = plt.gca()
    ax.set(xlabel="Sample Length", ylabel = "Time (LOG)", title= "Log Time comparison between sum_squares with for loop vs np.dot")
    ax.legend(['ssfor', 'ssnp'])
    plt.yscale('log')
    print(df)
    plt.show()
    plt.savefig('sumsquare.png')
    input()

    #timess_to_df(time_ss(gen_random_samples(10)))


if __name__ == "__main__":
    main()