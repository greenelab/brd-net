'''Compare the performance of different models in learning to differentiate between healthy
and unhealthy gene expression
'''
import argparse
import importlib
import os
import random
import sys

import numpy
import pandas
import sklearn
import tensorflow as tf

import classifier
import models
import utils

# Mute INFO and WARNING logs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_Z_files(Z_dir):
    '''Get the path to all files in the given directory that look like they were
    created by run_plier.R i.e. they have the extension '_Z.tsv'

    Arguments
    ---------
    Z_dir: str or Path
        The path to the directory to search

    Returns
    -------
    paths: list of strs
        The paths to each file in Z_dir
    '''
    files = [f for f in os.listdir(Z_dir) if f.endswith('_Z.tsv')]
    paths = [os.path.join(Z_dir, f) for f in files]
    return paths


def model_from_pyod(model_name):
    ''' Determine whether a model is from the PyOD module

    Arguments
    ---------
    model_name: str
        The name of the module containing the model, e.g. IForest or OCSVM.
        The naming convention of the PyOD package has the models' modules in
        lower case, while the model classes themselves are the same string
        but in upper camel case.

    Returns
    -------
    bool
        True if the model is found in pyod, false otherwise
    '''
    try:
        # Import the model module from pyod
        model_module = 'pyod.models.{}'.format(model_name.lower())
        module = importlib.import_module(model_module)
        getattr(module, model_name)

        return True
    except (AttributeError, ModuleNotFoundError):
        return False


def model_from_tf(model_name):
    ''' Determine whether a model is found in models.py

    Arguments
    ---------
    model_name: str
        The name of the model class in models.py

    Returns
    -------
    bool
        True if the model in found in models.py, false otherwise
    '''
    try:
        getattr(models, model_name)
        return True
    except AttributeError:
        return False


def eval_pyod_model(model_name, train_X, train_Y, val_X, val_Y, val_studies):
    '''Train a model from PyOD on train_X and train_Y, then evaluate its performance
    on val_X and val_Y

    Arguments
    ---------
    model_name: str
        The name of the model to be imported from PyOD
    train_X: numpy.array
        The gene expression data to train the model on
    train_Y: numpy.array
        The labels corresponding to whether each sample represents healthy or unhealthy
        gene expression
    val_X: numpy.array
        The gene expression data to be held out to evaluate model performance
    val_Y: numpy.array
        The labels corresponding to whether each sample in val_X represents healthy or
        unhealthy gene expression
    val_studies: list of strs
        The list containing which study each sample of val_X is from

    Returns
    -------
    val_acc: float
        The accuracy the model achieved in predicting val_Y from val_X
    val_auroc: float
        The area under the receiver operating characteristic curve based on the
        model's decision function on val_X
    val_aupr: float
        The area under the precision recall curve based on the model's decision function on val_X
    study_results: dict
        A dictionary mapping study ids to the model's performance on that study
    '''
    # Import the model module from pyod
    model_module = 'pyod.models.{}'.format(model_name.lower())
    module = importlib.import_module(model_module)
    model = getattr(module, model_name)

    model_instance = model(contamination=.5)
    model_instance.fit(train_X, train_Y)

    predictions = model_instance.predict(val_X)

    val_acc = utils.calculate_accuracy(predictions, val_Y)
    val_auroc = sklearn.metrics.roc_auc_score(val_Y, model_instance.decision_function(val_X))
    val_aupr = sklearn.metrics.average_precision_score(val_Y,
                                                       model_instance.decision_function(val_X))

    unique_studies = list(set(val_studies))

    study_results = {}
    for study in unique_studies:
        study_indices = [idx for idx, val_study in enumerate(val_studies) if study == val_study]
        study_data = val_X[study_indices]
        study_Y = val_Y[study_indices]

        # TODO this is identical to the eval code outside the for loop; refactor to a function
        predictions = model_instance.predict(study_data)
        val_acc = utils.calculate_accuracy(predictions, study_Y)

        try:
            raw_scores = model_instance.decision_function(study_data)
            val_auroc = sklearn.metrics.roc_auc_score(study_Y, raw_scores)
            val_aupr = sklearn.metrics.average_precision_score(study_Y, raw_scores)

            result_dict = {'val_loss': None, 'val_acc': val_acc,
                           'val_auroc': val_auroc, 'val_aupr': val_aupr}

            study_results[study] = result_dict
        except ValueError:
            sys.stderr.write('Warning: study {} only has one class, so auroc and aupr will not be '
                             'recorded\n'.format(study))
            result_dict = {'val_loss': None, 'val_acc': val_acc,
                           'val_auroc': None, 'val_aupr': None}
            study_results[study] = result_dict

    return val_acc, val_auroc, val_aupr, study_results


def train_tf_model(model_name, lr, train_X, train_Y, val_X, val_Y, checkpoint_dir,
                   logdir, epochs, seed):
    '''Train a model from models.py on train_X and train_Y, and write the best version
    to a file in checkpoint_dir

    Arguments
    ---------
    model_name: str
        The name of the model to be imported from models.py
    lr: float
        The size of each update step made by the optimizer
    train_X: numpy.array
        The gene expression data to train the model on
    train_Y: numpy.array
        The labels corresponding to whether each sample represents healthy or unhealthy
        gene expression
    val_X: numpy.array
        The gene expression data to be held out to evaluate model performance
    val_Y: numpy.array
        The labels corresponding to whether each sample in val_X represents healthy or
        unhealthy gene expression
    checkpoint_dir: str or Path
        The base directory in which to store checkpoint files for the best performing models
    logdir: str or Path or None
        The directory to save tensorboard logs to
    epochs: int
        The number of times the model should see the entirety of train_X before it completes
        training
    seed: int
        The current seed for the random number generator

    Returns
    -------
    val_acc: float
        The accuracy the model achieved in predicting val_Y from val_X
    val_auroc: float
        The area under the receiver operating characteristic curve based on the
        model's decision function on val_X
    '''
    lr_string = '{:.0e}'.format(lr)
    checkpoint_string = '{}_{}_{}'.format(model_name, lr_string, seed)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_string, 'checkpoint')

    os.makedirs(os.path.join(checkpoint_dir, checkpoint_string), exist_ok=True)

    classifier.train_model(train_X, train_Y, val_X, val_Y, checkpoint_path,
                           model_name=model_name, lr=lr,
                           epochs=int(epochs))

    return checkpoint_path


def eval_tf_model(model_name, lr, val_X, val_Y, checkpoint_path, logdir):
    ''' Evaluate the performance of the model saved at checkpoint_path on the full dataset

    Arguments
    ---------
    model_name: str
        The name of the model to be imported from models.py
    lr: float
        The size of each update step made by the optimizer
    val_X: numpy.array
        The gene expression data to be held out to evaluate model performance
    val_Y: numpy.array
        The labels corresponding to whether each sample in val_X represents healthy or
        unhealthy gene expression
    checkpoint_path: str
        The path to the weights for the best performing iteration of the model
    logdir: str or Path or None
        The directory to save tensorboard logs to

    Returns
    -------
    val_acc: float
        The accuracy the model achieved in predicting val_Y from val_X
    val_auroc: float
        The area under the receiver operating characteristic curve based on the
        model's decision function on val_X
    '''
    # Load model
    model = utils.get_model(model_name, logdir, lr)
    model.load_weights(checkpoint_path)

    val_loss, val_acc, val_auroc, val_aupr = model.evaluate(val_X, val_Y)
    return val_acc, val_auroc, val_aupr


def eval_tf_model_studies(model_name, lr, val_X, val_Y, val_studies, checkpoint_path, logdir):
    ''' Evaluate the performance of the model saved at checkpoint_path on each study individually

    Arguments
    ---------
    model_name: str
        The name of the model to be imported from models.py
    lr: float
        The size of each update step made by the optimizer
    val_X: numpy.array
        The gene expression data to be held out to evaluate model performance
    val_Y: numpy.array
        The labels corresponding to whether each sample in val_X represents healthy or
        unhealthy gene expression
    val_studies: list of strs
        The list containing which study each sample of val_X is from
    checkpoint_path: str
        The path to the weights for the best performing iteration of the model
    logdir: str or Path or None
        The directory to save tensorboard logs to

    Returns
    -------
    study_to_metrics: dict
        A dictionary mapping study ids to the model's performance on that study
    '''
    model = utils.get_model(model_name, logdir, lr)
    model.load_weights(checkpoint_path)

    unique_studies = list(set(val_studies))

    study_to_metrics = {}
    for study in unique_studies:
        study_indices = [idx for idx, val_study in enumerate(val_studies) if study == val_study]
        study_data = val_X[study_indices]
        study_Y = val_Y[study_indices]

        val_loss, val_acc, val_auroc, val_aupr = model.evaluate(study_data, study_Y)
        result_dict = {'val_loss': val_loss, 'val_acc': val_acc,
                       'val_auroc': val_auroc, 'val_aupr': val_aupr}
        study_to_metrics[study] = result_dict

    return study_to_metrics


def update_study_losses_from_study_results(study_results, study_losses, model, lr, seed,
                                           latent_var_count):
    ''' Create a tuple corresponding to model results for multiple studies, and store the
    tuple in study_losses

    Arguments
    ---------
    study_results: dict
        A dictionary mapping study ids to the model's performance on that study
    study_losses: list of tuples
        The results from all models run so far calculated per study
    model: str
        The name of the model to be imported from models.py
    lr: float
        The size of each update step made by the optimizer
    seed: int
        The current seed for the random number generator
    latent_var_count: int
        The number of latent variabes to be used by PLIER (the number of PLIER PCs)

    Returns
    -------
    '''
    for study in study_results:
        study_indices = [idx for idx, val_study in enumerate(val_studies) if study == val_study]
        study_Y = val_Y[study_indices]
        study_baseline = utils.get_larger_class_percentage(study_Y)

        val_acc = study_results[study]['val_acc']
        val_auroc = study_results[study]['val_auroc']
        val_aupr = study_results[study]['val_aupr']
        study_losses.append((model, study, lr, seed, val_acc,
                             val_auroc, val_aupr, study_baseline,
                             latent_var_count))
    return study_losses


def train_and_evaluate_model(model, lr, train_X, train_Y, val_X, val_Y, val_studies,
                             checkpoint_dir, logdir, epochs, seed, losses, study_losses):
    ''' Train and evaluate the given model with the given hyperparametrs.

    Arguments
    ---------
    model: str
        The name of the model to be imported from models.py
    lr: float
        The size of each update step made by the optimizer
    train_X: numpy.array
        The gene expression data to train the model on
    train_Y: numpy.array
        The labels corresponding to whether each sample represents healthy or unhealthy
        gene expression
    val_X: numpy.array
        The gene expression data to be held out to evaluate model performance
    val_Y: numpy.array
        The labels corresponding to whether each sample in val_X represents healthy or
        unhealthy gene expression
    val_studies: list of strs
        The list containing which study each sample of val_X is from
    checkpoint_dir: str or Path
        The base directory in which to store checkpoint files for the best performing models
    logdir: str or Path or None
        The directory to save tensorboard logs to
    epochs: int
        The number of times the model should see the entirety of train_X before it completes
        training
    seed: int
        The current seed for the random number generator
    losses: list of tuples
        The results from all models run so far aggregated across all studies
    study_losses: list of tuples
        The results from all models run so far calculated per study
    '''
    val_acc = None
    val_auroc = None
    val_aupr = None
    study_results = None
    latent_var_count = train_X.shape[1]

    if model_from_tf(model):
        path_to_model = train_tf_model(model, lr, train_X, train_Y, val_X, val_Y, checkpoint_dir,
                                       logdir, epochs, seed)

        val_acc, val_auroc, val_aupr = eval_tf_model(model, lr, val_X, val_Y, path_to_model,
                                                     logdir)
        study_results = eval_tf_model_studies(model, lr, val_X, val_Y, val_studies, path_to_model,
                                              logdir)

    if model_from_pyod(model):
        metrics = eval_pyod_model(model, train_X, train_Y, val_X, val_Y,
                                  val_studies)
        val_acc, val_auroc, val_aupr, study_results = metrics

    losses.append((model, lr, seed, val_acc, val_auroc, val_aupr,
                   val_baseline, latent_var_count))

    study_losses = update_study_losses_from_study_results(study_results, study_losses, model, lr,
                                                          seed, latent_var_count)

    return losses, study_losses


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser()

    parser.add_argument('Z_file_dir', help='Path to the directory containing '
                                           'PLIER matrices to be used to convert '
                                           'the data into LV space')
    parser.add_argument('healthy_file_path', help='Path to the tsv containing healthy gene '
                                                  'expression')
    parser.add_argument('disease_file_path', help='Path to the tsv containing unhealthy '
                                                  'gene expression data')
    parser.add_argument('--epochs', help='The maximum number of epochs to train the model for',
                        default=400)
    parser.add_argument('--checkpoint_dir', help='The directory to save model weights to',
                        default='{}/../checkpoints'.format(script_dir))
    parser.add_argument('--logdir', help='The directory to log training progress to')
    parser.add_argument('--out_path', help='The file to print the csv containing the results to',
                        default='{}/../results/model_eval_results.csv'.format(script_dir))
    parser.add_argument('--study_out_path', help='The file to print the csv containing results '
                                                 'aggregated at the study level',
                        default='{}/../results/model_eval_study_results.csv'.format(script_dir))
    parser.add_argument('--num_seeds', help='The number of times to randomly select a '
                        'validation dataset', default=10, type=int)
    parser.add_argument('--learning_rates', help='The learning rate or rates to use for each '
                        'tensorflow model', nargs='*', type=float, default=[1e-5])
    parser.add_argument('--pyod_models', help='The case sensitive list of names of PyOD'
                        'models to evaluate. A list of all possible names can be found at'
                        'https://pyod.readthedocs.io/en/latest/pyod.html',
                        nargs='*', type=str, default=['IForest', 'OCSVM'])

    args = parser.parse_args()

    # Add the names of pyod models manually, as there are too many to iterate over all of them
    model_list = []
    if args.pyod_models is not None:
        model_list = args.pyod_models
    model_list.extend(utils.get_model_list())

    Z_files = get_Z_files(args.Z_file_dir)

    losses = []
    study_losses = []
    try:
        for Z_file_path in Z_files:
            for seed in range(args.num_seeds):
                # Set random seeds
                random.seed(seed)
                numpy.random.seed(seed)
                tf.compat.v1.set_random_seed(seed)

                data = utils.load_data_and_studies(Z_file_path, args.healthy_file_path,
                                                   args.disease_file_path)
                train_X, train_Y, val_X, val_Y, train_studies, val_studies = data

                val_baseline = utils.get_larger_class_percentage(val_Y)
                for lr in args.learning_rates:
                    for model in model_list:
                        model_params = (model, lr, train_X, train_Y, val_X, val_Y, val_studies,
                                        args.checkpoint_dir, args.logdir, args.epochs, seed,
                                        losses, study_losses)
                        losses, study_losses = train_and_evaluate_model(*model_params)

    finally:
        # If there is an error somewhere in the training process, save the results so far
        results_df = pandas.DataFrame.from_records(losses, columns=['Model',
                                                                    'LR',
                                                                    'Seed',
                                                                    'val_acc',
                                                                    'val_auroc',
                                                                    'val_aupr',
                                                                    'val_baseline',
                                                                    'lv_count',
                                                                    ])
        study_results_df = pandas.DataFrame.from_records(study_losses, columns=['Model',
                                                                                'Study',
                                                                                'LR',
                                                                                'Seed',
                                                                                'val_acc',
                                                                                'val_auroc',
                                                                                'val_aupr',
                                                                                'val_baseline',
                                                                                'lv_count',
                                                                                ])

        results_df.to_csv(args.out_path)
        study_results_df.to_csv(args.study_out_path)
