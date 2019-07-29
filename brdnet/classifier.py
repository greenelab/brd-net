
import argparse

import numpy
import pandas
import tensorflow as tf

import models

# Set random seeds
numpy.random.seed(42)
tf.compat.v1.set_random_seed(42)


def get_model(model_name, logdir):
    '''Retrieve a Model object from the models.py module by name

    Arguments
    ---------
    model_name: string
        The name of the model to retrieve
    logdir: string
        The path to the directory to save logs to

    Returns
    -------
    model: Model
        The model object with name model_name
    '''
    # This retrieves whatever has the name model_name in the models module
    model = getattr(models, model_name)

    modelInstance = model()

    optimizer = tf.keras.optimizers.Adam(lr=1e-6)

    modelInstance.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'],
                         )
    return modelInstance


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

    # Ensure the gene symbols are in alphabetical order
    Z_df = Z_df.sort_index()

    healthy_df = pandas.read_csv(args.healthy_file_path, sep='\t')
    healthy_matrix = reduce_dimensionality(healthy_df, Z_df)

    disease_df = pandas.read_csv(args.disease_file_path, sep='\t')
    disease_matrix = reduce_dimensionality(disease_df, Z_df)

    healthy_labels = numpy.zeros(healthy_df.shape[1])
    disease_labels = numpy.ones(disease_df.shape[1])

    train_X = numpy.concatenate([healthy_matrix, disease_matrix])
    train_Y = numpy.concatenate([healthy_labels, disease_labels])

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
    parser.add_argument('--logdir', help='The directory to print tensorboard logs to',
                        default='../logs')

    args = parser.parse_args()
    print(args.logdir)


    train_X, train_Y = load_data(args)

    model = get_model('MLP', args.logdir)

    model.fit(train_X, train_Y,
            batch_size=16,
            epochs=1000,
            validation_split=.2,
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir='../logs')]
            )
