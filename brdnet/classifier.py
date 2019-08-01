'''Train models to differentiate between case and control gene expression data'''

import argparse
from collections import Counter
import inspect
import random
import sys
import time
import os

import numpy
import pandas
import tensorflow as tf

import models

def get_model_list():
    '''Return the list of model classes in the models module as a list of strings'''
    model_list = []

    # The only classes that should be in models.py are models, so we can
    # iterate over all the classes in the module to get which models exist
    for model in inspect.getmembers(models, inspect.isclass):
        model_list.append(model[0])

    return model_list


def write_invalid_model_error(model_name):
    '''Write the error message for an invalid model, model_name'''
    sys.stderr.write('Error: models.py does not contain the model {}\n'.format(model_name))
    sys.stderr.write('The available models are:\n')
    sys.stderr.write('\n'.join(get_model_list()))


def validate_model_name(model_name):
    '''Check whether the model name supplied by the user exists in models.py'''
    try:
        model = getattr(models, model_name)
    except AttributeError:
        write_invalid_model_error(model_name)
        sys.exit()


def get_model(model_name, logdir, lr):
    '''Retrieve a Model object from the models.py module by name

    Arguments
    ---------
    model_name: string
        The name of the model to retrieve
    logdir: string
        The path to the directory to save logs to
    lr: float
        The learning rate to be used by the optimizer

    Returns
    -------
    model: Model
        The model object with name model_name
    '''
    # This retrieves whatever has the name model_name in the models module
    model = getattr(models, model_name)

    modelInstance = model()

    optimizer = tf.keras.optimizers.Adam(lr=lr)

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
    # the two dataframes, we can drop the labels and create a numpy matrix to be multiplied by Z
    expression_matrix = expression_df.values
    Z_matrix = Z_df.values

    reduced_matrix = numpy.matmul(expression_matrix.T, Z_matrix)

    return reduced_matrix


def get_validation_set(data, validation_fraction=.2):
    '''Split a dataframe into training and validation data by extracting studies
       that contain a certain fraction of the samples

       Arguments
       ---------
       data: pandas.DataFrame
           data is a dataframe where the rows represent genes and columns represent samples.
           The column names should be of the format 'studyid.runid' to allow the study
           information to be used.
       validation_fraction: float
           The fraction of the dataset to be pulled out as validation data

       Returns
       -------
       train_df: pandas.DataFrame
           A dataframe containing the fraction of the sample to be used in training
       val_df:
           A dataframe containing the fraction of the sample to be used for validation
    '''
    studies = [name.split('.')[0] for name in data.columns]
    # random.shuffle shuffles in place
    random.shuffle(studies)

    counter = Counter(studies)

    target_sample_count = len(data.columns) * validation_fraction

    samples_so_far = 0
    val_studies = []
    for study, samples in counter.items():
        # Prevent the validation set from being too much larger than the target
        if samples_so_far + samples > len(data.columns) * (validation_fraction + .05):
            continue
        val_studies.append(study)
        samples_so_far += samples

        if samples_so_far >= target_sample_count:
            break

    val_columns = []
    for study in val_studies:
        val_columns.extend([col for col in data.columns if study in col])

    val_df = data[val_columns]
    train_df = data[data.columns.difference(val_columns)]

    return train_df, val_df


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

    healthy_train, healthy_val = get_validation_set(healthy_df)
    sys.exit(0)

    #TODO reduce dimensionality of both train and val dfs
    healthy_matrix = reduce_dimensionality(healthy_df, Z_df)

    disease_df = pandas.read_csv(args.disease_file_path, sep='\t')
    disease_matrix = reduce_dimensionality(disease_df, Z_df)

    healthy_labels = numpy.zeros(healthy_df.shape[1])
    disease_labels = numpy.ones(disease_df.shape[1])

    train_X = numpy.concatenate([healthy_matrix, disease_matrix])
    train_Y = numpy.concatenate([healthy_labels, disease_labels])

    return train_X, train_Y


if __name__ == '__main__':
    model_list = get_model_list()

    timestamp = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('Z_file_path', help='Path to the PLIER matrix to be used to convert '
                                            'the data into LV space')
    parser.add_argument('healthy_file_path', help='Path to the tsv containing healthy gene '
                                                  'expression')
    parser.add_argument('disease_file_path', help='Path to the tsv containing unhealthy '
                                                  'gene expression data')
    parser.add_argument('--logdir', help='The directory to print tensorboard logs to',
                        default='../logs/{}'.format(timestamp))
    parser.add_argument('--model', help='The name of the model to be used. The models currently '
                        'available are: {}'.format(', '.join(model_list)), default='MLP')
    parser.add_argument('-s', '--seed', help='The seed to be used in random number generators', default=42)
    parser.add_argument('-l', '--learning_rate', help='The learning rate to be used by the '
                                                      'optimization process',
                        default=1e-5)
    parser.add_argument('--epochs', help='The maximum number of epochs to train the model for',
                        default=1000)

    args = parser.parse_args()

    # Set random seeds
    numpy.random.seed(args.seed)
    tf.compat.v1.set_random_seed(args.seed)

    # Create log directory
    os.mkdir(args.logdir)

    model_name = args.model
    validate_model_name(model_name)

    train_X, train_Y = load_data(args)

    model = get_model(model_name, args.logdir, args.learning_rate)

    model.fit(train_X, train_Y,
            batch_size=16,
            epochs=args.epochs,
            validation_split=.2,
            callbacks=[tf.keras.callbacks.TensorBoard(log_dir=args.logdir),
                       tf.keras.callbacks.ReduceLROnPlateau(min_lr=1e-11),
                      ])
