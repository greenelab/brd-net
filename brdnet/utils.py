from collections import Counter
import inspect
import random

import numpy
import pandas
import tensorflow as tf

import models


def reduce_dimensionality(expression_df, Z_df):
    '''Convert a dataframe of gene expression data from gene space to the low
    dimensional representation specified by Z_df

    Arguments
    ---------
    expression_df: pandas.DataFrame
        The expression dataframe to transform to the low dimensional representation
        specified by Z_df
    Z_df: pandas.dataframe
        The matrix that does the conversion from gene space to the low
        dimensional representation specified by Z_df

    Returns
    -------
    reduced_matrix: numpy.array
        The result from translating expression_df into the low dimensional representation
        specified by Z_df
    '''

    # Don't sort the genes in Z_df if they're already sorted
    if not Z_df.index.is_monotonic:
        # Ensure the gene symbols are in alphabetical order
        Z_df = Z_df.sort_index()

    expression_df = expression_df[expression_df.index.isin(Z_df.index)]

    expression_df = expression_df.sort_index()

    # Since the gene symbols are in alphabetical order and are identical between
    # the two dataframes, we can drop the labels and create a numpy matrix to be multiplied by Z
    expression_matrix = expression_df.values
    Z_matrix = Z_df.values

    reduced_matrix = numpy.matmul(expression_matrix.T, Z_matrix)

    return reduced_matrix


def get_study_list(expression_df):
    '''Retrieve the studies for each sample in the provided dataframe

    Arguments
    ---------
    expression_df: pandas.DataFrame
        The dataframe containing gene expression. Assumes the columns of the dataframe are named
        in study.sample format

    Returns
    -------
    studies: list of strs
        The list containing the study each sample is from
    '''
    samples = expression_df.columns
    studies = [sample.split('.')[0] for sample in samples]

    return studies


def prepare_input_data(Z_df, healthy_df, disease_df):
    '''Convert the dataframes from run_plier and download_categorized_data into
    training and validation datasets with accompanying labels

    Arguments
    ---------
    Z_df: pandas.DataFrame
        The matrix to convert the expression data into a lower dimensional representation
    healthy_df: pandas.DataFrame
        The dataframe containing healthy gene expression samples
    disease_df: pandas.DataFrame
        The dataframe containin unhealthy gene expression samples

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
    train_studies: list of strs
        The list containing which study each sample of train_X is from
    val_studies: list of strs
        The list containing which study each sample of val_X is from
    '''
    # TODO stop val fraction from being hardcoded
    healthy_train, healthy_val, disease_train, disease_val = get_validation_set(healthy_df,
                                                                                disease_df,
                                                                                .2)
    healthy_train_studies = get_study_list(healthy_train)
    healthy_val_studies = get_study_list(healthy_val)
    disease_train_studies = get_study_list(disease_train)
    disease_val_studies = get_study_list(disease_val)

    healthy_train = reduce_dimensionality(healthy_train, Z_df)
    healthy_val = reduce_dimensionality(healthy_val, Z_df)
    disease_train = reduce_dimensionality(disease_train, Z_df)
    disease_val = reduce_dimensionality(disease_val, Z_df)

    healthy_train_labels = numpy.zeros(healthy_train.shape[0])
    healthy_val_labels = numpy.zeros(healthy_val.shape[0])
    disease_train_labels = numpy.ones(disease_train.shape[0])
    disease_val_labels = numpy.ones(disease_val.shape[0])

    healthy_train_studies.extend(disease_train_studies)
    healthy_val_studies.extend(disease_val_studies)
    # Rename the variables to make the fact that extend runs in place less confusing
    train_studies = healthy_train_studies
    val_studies = healthy_val_studies

    train_X = numpy.concatenate([healthy_train, disease_train])
    train_Y = numpy.concatenate([healthy_train_labels, disease_train_labels])

    val_X = numpy.concatenate([healthy_val, disease_val])
    val_Y = numpy.concatenate([healthy_val_labels, disease_val_labels])

    return train_X, train_Y, val_X, val_Y, train_studies, val_studies


def load_data_and_studies(Z_file_path, healthy_file_path, disease_file_path):
    '''Load and process the training data

    Arguments
    ---------
    Z_file_path: str or Path object
        The path to the file containing the dataframe for transforming expression data
        into a low dimensional representation
    healthy_file_path: str or Path object
        The path to the file containing healthy gene expression data
    disease_file_path: str or Path object
        The path to the file containing unhealthy gene expression data

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
    train_studies: list of strs
        The list containing which study each sample of train_X is from
    val_studies: list of strs
        The list containing which study each sample of val_X is from
    '''
    Z_df = pandas.read_csv(Z_file_path, sep='\t')

    healthy_df = pandas.read_csv(healthy_file_path, sep='\t')
    disease_df = pandas.read_csv(disease_file_path, sep='\t')

    return prepare_input_data(Z_df, healthy_df, disease_df)


def get_larger_class_percentage(Y):
    '''Calculate the percentage of the labels that belong to the largest class

    Arguments
    ---------
    Y: numpy array
        The labels for the data

    Returns
    -------
    percentage: float
        The percentage of the labels that belongs to the largest class
    '''
    _, counts = numpy.unique(Y, return_counts=True)

    return max(counts) / sum(counts)


def calculate_accuracy(pred_Y, true_Y):
    '''Calculate the accuracy for a set of predicted classification labels'''
    # We use subtraction and count_nonzero because logical_xor only works for binary labels
    num_incorrect = numpy.count_nonzero(numpy.subtract(pred_Y, true_Y))
    acc = (len(pred_Y) - num_incorrect) / len(pred_Y)
    return acc


def get_model_list():
    '''Return the list of model classes in the models module as a list of strings'''
    model_list = []

    # The only classes that should be in models.py are models, so we can
    # iterate over all the classes in the module to get which models exist
    for model in inspect.getmembers(models, inspect.isclass):
        model_list.append(model[0])

    return model_list


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

    model_instance = model()

    optimizer = tf.keras.optimizers.Adam(lr=lr)

    auroc = tf.keras.metrics.AUC(curve='ROC')
    auprc = tf.keras.metrics.AUC(curve='PR')

    model_instance.compile(optimizer=optimizer,
                           loss='binary_crossentropy',
                           metrics=['accuracy', auroc, auprc],
                           )
    return model_instance


def get_study_counter(df):
    '''Get the number of samples belonging to each study in a given pandas DataFrame'''
    studies = [name.split('.')[0] for name in df.columns]

    counter = Counter(studies)

    return counter


def get_validation_set(healthy_df, disease_df, validation_fraction=.2):
    '''Split a dataframe into training and validation data by extracting studies
       that contain a certain fraction of the samples

       Arguments
       ---------
       healthy_df: pandas.DataFrame
           A dataframe where the rows represent genes and columns represent samples.
           The column names should be of the format 'studyid.runid' to allow the study
           information to be used. healthy_df contains samples with healthy gene expression
       disease_df: pandas.DataFrame
           A dataframe where the rows represent genes and columns represent samples.
           The column names should be of the format 'studyid.runid' to allow the study
           information to be used. disease_df contains samples with unhealthy gene expression
       validation_fraction: float
           The fraction of the dataset to be pulled out as validation data

       Returns
       -------
       train_df: pandas.DataFrame
           A dataframe containing the fraction of the sample to be used in training
       val_df:
           A dataframe containing the fraction of the sample to be used for validation
    '''
    healthy_counter = get_study_counter(healthy_df)
    disease_counter = get_study_counter(disease_df)

    healthy_target_count = len(healthy_df.columns) * validation_fraction
    disease_target_count = len(disease_df.columns) * validation_fraction

    healthy_samples_so_far = 0
    disease_samples_so_far = 0
    val_studies = []

    all_studies = set(healthy_counter.keys())
    all_studies.update(disease_counter.keys())
    all_studies = list(all_studies)
    random.shuffle(all_studies)

    for study in all_studies:
        healthy_samples = healthy_counter[study]
        disease_samples = disease_counter[study]

        # Prevent the validation set from being too much larger than the target
        if (healthy_samples + healthy_samples_so_far > healthy_target_count * 1.05
                or disease_samples + disease_samples_so_far > disease_target_count * 1.05):
            continue

        val_studies.append(study)
        healthy_samples_so_far += healthy_samples
        disease_samples_so_far += disease_samples

        if (healthy_samples_so_far >= healthy_target_count
                and disease_samples_so_far >= disease_target_count):
            break

    healthy_train, healthy_val = get_val_and_train_subset(healthy_df, val_studies)
    disease_train, disease_val = get_val_and_train_subset(disease_df, val_studies)

    return healthy_train, healthy_val, disease_train, disease_val


def get_val_and_train_subset(df, val_studies):
    ''' Get the subset of a dataframe whose column names contain the strings in val_studies

    Arguments
    ---------
    df: pandas.DataFrame
        The dataframe to be split
    val_studies: list of str
        The names of the studies that should go in the validation subset

    Returns
    -------
    train_df: pandas.DataFrame
        The dataframe containing the fraction of df to be used as training data
    val_df: pandas.DataFrame
        The dataframe containing the fraction of df to be used as validation data
    '''
    val_columns = []
    for study in val_studies:
        val_columns.extend([col for col in df.columns if study in col])

    val_df = df[val_columns]
    train_df = df[df.columns.difference(val_columns)]

    return train_df, val_df
