'''Train models to differentiate between case and control gene expression data'''

import argparse
import random
import sys
import time
import os

import numpy
import pandas
import tensorflow as tf

import models
import utils

def write_invalid_model_error(model_name):
    '''Write the error message for an invalid model, model_name'''
    sys.stderr.write('Error: models.py does not contain the model {}\n'.format(model_name))
    sys.stderr.write('The available models are:\n')
    sys.stderr.write('\n'.join(utils.get_model_list()))


def validate_model_name(model_name):
    '''Check whether the model name supplied by the user exists in models.py'''
    try:
        model = getattr(models, model_name)
    except AttributeError:
        write_invalid_model_error(model_name)
        sys.exit()


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
    # the two dataframes, we can drop the labels and create a numpy matrix to be multiplied by Z
    expression_matrix = expression_df.values
    Z_matrix = Z_df.values

    reduced_matrix = numpy.matmul(expression_matrix.T, Z_matrix)

    return reduced_matrix


def prepare_input_data(Z_df, healthy_df, disease_df, random_seed):
    '''Convert the dataframes from run_plier and download_categorized_data into
    training and validation datasets with accompanying labels

    Arguments
    ---------
    Z_df: pandas.DataFrame
        The matrix to convert the expression data into the PLIER latent space
    healthy_df: pandas.DataFrame
        The dataframe containing healthy gene expression samples
    disease_df: pandas.DataFrame
        The dataframe containin unhealthy gene expression samples
    random_seed: int
        The seed to be used by utils.get_validation_set

    Returns
    -------
    train_X: numpy.array
        A numpy array containing the training gene expression data
    train_Y: numpy.array
        The labels corresponding to whether each sample represents healthy or unhealthy
        gene expression
    val_X: numpy.array
        The gene expression data to be held out to evaluate model training
    val_Y: numpy.array
        The labels for val_X
    '''
    healthy_train, healthy_val, disease_train, disease_val = utils.get_validation_set(healthy_df,
                                                                                      disease_df,
                                                                                      .2,
                                                                                      random_seed)
    healthy_train = reduce_dimensionality(healthy_train, Z_df)
    healthy_val = reduce_dimensionality(healthy_val, Z_df)
    disease_train = reduce_dimensionality(disease_train, Z_df)
    disease_val = reduce_dimensionality(disease_val, Z_df)

    healthy_train_labels = numpy.zeros(healthy_train.shape[0])
    healthy_val_labels = numpy.zeros(healthy_val.shape[0])
    disease_train_labels = numpy.ones(disease_train.shape[0])
    disease_val_labels = numpy.ones(disease_val.shape[0])

    train_X = numpy.concatenate([healthy_train, disease_train])
    train_Y = numpy.concatenate([healthy_train_labels, disease_train_labels])

    val_X = numpy.concatenate([healthy_val, disease_val])
    val_Y = numpy.concatenate([healthy_val_labels, disease_val_labels])

    return train_X, train_Y, val_X, val_Y


def load_data(Z_file_path, healthy_file_path, disease_file_path, seed):
    '''Load and process the training data

    Arguments
    ---------
    Z_file_path: str or Path object
        The path to the file containing the dataframe for moving expression data
        into the PLIER latent space
    healthy_file_path: str or Path object
        The path to the file containing healthy gene expression data
    disease_file_path: str or Path object
        The path to the file containing unhealthy gene expression data
    seed: int
        The random seed to be used in sampling validation data

    Returns
    -------
    train_X: numpy.array
        A numpy array containing the training gene expression data
    train_Y: numpy.array
        The labels corresponding to whether each sample represents healthy or unhealthy
        gene expression
    val_X: numpy.array
        The gene expression data to be held out to evaluate model training
    val_Y: numpy.array
        The labels corresponnding to whether each sample in val_X represents healthy
        or unhealthy gene expression
    '''
    Z_df = pandas.read_csv(Z_file_path, sep='\t')

    # Ensure the gene symbols are in alphabetical order
    Z_df = Z_df.sort_index()

    healthy_df = pandas.read_csv(healthy_file_path, sep='\t')
    disease_df = pandas.read_csv(disease_file_path, sep='\t')

    return prepare_input_data(Z_df, healthy_df, disease_df, seed)


def train_model(train_X, train_Y, val_X, val_Y, checkpoint_path,
                model_name, logdir=None, lr=1e-6, epochs=1000, batch_size=16,
                ):

    validate_model_name(model_name)

    model = utils.get_model(model_name, logdir, lr)


    callbacks = [tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-11),
                 tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    save_weights_only=True,
                                                    save_best_only=True)
                ]
    # Create log directory
    if logdir is not None:
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir),
                     tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-11),
                     tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                        save_weights_only=True,
                                                        save_best_only=True)
                    ]

    model.fit(train_X, train_Y,
              callbacks=callbacks,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(val_X, val_Y),
              verbose=0,
              )
    return model


if __name__ == '__main__':
    model_list = utils.get_model_list()

    timestamp = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('Z_file_path', help='Path to the PLIER matrix to be used to convert '
                                            'the data into LV space')
    parser.add_argument('healthy_file_path', help='Path to the tsv containing healthy gene '
                                                  'expression')
    parser.add_argument('disease_file_path', help='Path to the tsv containing unhealthy '
                                                  'gene expression data')
    parser.add_argument('--logdir', help='The directory to print tensorboard logs to')
    parser.add_argument('--model', help='The name of the model to be used. The models currently '
                        'available are: {}'.format(', '.join(model_list)), default='MLP')
    parser.add_argument('-s', '--seed', help='The seed to be used in random number generators',
                        default=42)
    parser.add_argument('-l', '--learning_rate', help='The base learning rate to be used by the '
                                                      'optimization process. When the validation '
                                                      'loss reaches a plateau, the learning rate '
                                                      'will decrease until it reaches 1e-11',
                        default=1e-5)
    parser.add_argument('--epochs', help='The maximum number of epochs to train the model for',
                        default=1000)
    parser.add_argument('--batch_size', help='The number of training samples in a batch',
                        default=16)
    parser.add_argument('--checkpoint_dir', help='The directory to save model weights to',
                        default='../checkpoints')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    numpy.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    train_X, train_Y, val_X, val_Y = load_data(args)

    validate_model_name(args.model)

    full_checkpoint_dir = os.path.join(args.checkpoint_dir, args.model)
    if not os.path.isdir(full_checkpoint_dir):
        os.makedirs(full_checkpoint_dir)

    checkpoint_path = os.path.join(full_checkpoint_dir, 'checkpoint')

    train_model(train_X, train_Y, val_X, val_Y, checkpoint_path,
                args.model, args.logdir, args.learning_rate, args.epochs, args.batch_size)
