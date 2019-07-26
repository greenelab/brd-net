
import argparse

import numpy
import pandas
#import tensorflow as tf



def reduce_dimensionality(expression_df, Z_df):
    '''Convert a dataframe of gene expression data from gene space to LV space

    Arguments
    ---------
    expression_df: pandas.DataFrame
        The expression dataframe to move to LV space
    Z_df: pandas.dataframe
        The matrix that does the conversion from gene space to LV space

    Returns
    -------
    reduced_matrix: numpy.array
        The result from translating expression_df into LV space
    '''
    expression_df = expression_df[expression_df.index.isin(Z_df.index)]

    expression_df = expression_df.sort_index()

    # Since the gene symbols are in alphabetical order and are identical between
    # the two dataframes, we can drop the labels and create a numpy matrix to be multiplier by Z
    expression_matrix = expression_df.values
    Z_matrix = Z_df.values

    print(Z_matrix.shape, expression_matrix.shape)

    reduced_matrix = numpy.matmul(expression_matrix.T, Z_matrix)

    return reduced_matrix


def load_data(args):
    '''Load and process the training data

    Arguments
    ---------
    args: namespace
        The command line arguments passed in to the script

    Returns
    -------
    train_X: numpy.array
        A numpy array containing the training gene expression data
    train_Y: numpy.array
        The labels corresponding to whether each sample represents healthy or unhealthy
        gene expression
    '''
    Z_df = pandas.read_csv(args.Z_file_path, sep='\t')
    print(Z_df.head())
    # Ensure the gene symbols are in alphabetical order
    Z_df = Z_df.sort_index()

    healthy_df = pandas.read_csv(args.healthy_file_path, sep='\t')
    healthy_matrix = reduce_dimensionality(healthy_df, Z_df)

    disease_df = pandas.read_csv(args.disease_file_path, sep='\t')
    disease_matrix = reduce_dimensionality(disease_df, Z_df)

    healthy_labels = numpy.zeros(healthy_df.shape[1])
    disease_labels = numpy.ones(disease_df.shape[1])

    print(healthy_labels.shape)

    train_X = numpy.concatenate([healthy_matrix, disease_matrix])
    train_Y = numpy.concatenate([healthy_labels, disease_labels])

    print(train_X.shape)
    print(train_Y.shape)

    #TODO keep metadata with training data somehow

    return train_X, train_Y


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('Z_file_path', help='Path to the PLIER matrix to be used to convert '
                                            'the data into LV space')
    parser.add_argument('healthy_file_path', help='Path to the tsv containing healthy gene '
                                                  'expression')
    parser.add_argument('disease_file_path', help='Path to the tsv containing unhealthy '
                                                  'gene expression data')

    args = parser.parse_args()

    train_X, train_Y = load_data(args)



    # Create training data pipeline (1 sample/batch at a time? Split matrix?)
    # Create validation set

    # load model architecture

    # Train model

    # Evaluate model


    # Where should final evaluation data come from?!?
