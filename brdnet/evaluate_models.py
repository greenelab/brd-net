'''Compare the performance of different models in learning to differentiate between healthy
and unhealthy gene expression
'''
import argparse
import importlib
import os

import pandas
import sklearn

import classifier
import models
import utils

# Mute INFO and WARNING logs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_Z_files(Z_dir):
    '''Get the path to all files in the given directory that look like they were
    created by run_plier.R

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


def eval_pyod_model(model_name, train_X, train_Y, val_X, val_Y):
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

    Returns
    -------
    val_acc: float
        The accuracy the model achieved in predicting val_Y from val_X
    val_auroc: float
        The area under the receiver operating characteristic curve based on the
        model's decision function on val_X
    '''
    # Import the model module from pyod
    model_module = 'pyod.models.{}'.format(model_name.lower())
    module = importlib.import_module(model_module)
    model = getattr(module, model_name)

    model_instance = model()
    model_instance.fit(train_X, train_Y)

    predictions = model_instance.predict(val_X)

    val_acc = utils.calculate_accuracy(predictions, val_Y)
    val_auroc = sklearn.metrics.roc_auc_score(val_Y, model_instance.decision_function(val_X))

    return val_acc, val_auroc


def eval_tf_model(model_name, lr, train_X, train_Y, val_X, val_Y, checkpoint_dir,
                  logdir, epochs, seed):
    '''Train a model from models.py on train_X and train_Y, then evaluate its performance
    on val_X and val_Y

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

    # Load model
    untrained_model = utils.get_model(model, logdir, lr)

    val_loss, val_acc, val_auroc = untrained_model.evaluate(val_X, val_Y)

    untrained_model.load_weights(checkpoint_path)
    val_loss, val_acc, val_auroc = untrained_model.evaluate(val_X, val_Y)
    return val_acc, val_auroc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('Z_file_dir', help='Path to the directory containing '
                                           'PLIER matrices to be used to convert '
                                           'the data into LV space')
    parser.add_argument('healthy_file_path', help='Path to the tsv containing healthy gene '
                                                  'expression')
    parser.add_argument('disease_file_path', help='Path to the tsv containing unhealthy '
                                                  'gene expression data')
    parser.add_argument('--epochs', help='The maximum number of epochs to train the model for',
                        default=1000)
    parser.add_argument('--checkpoint_dir', help='The directory to save model weights to',
                        default='../checkpoints')
    parser.add_argument('--logdir', help='The directory to log training progress to')
    parser.add_argument('--out_path', help='The file to print the csv containing the results to',
                        default='../results/model_eval_results.csv')
    parser.add_argument('--num_seeds', help='The number of times to randomly select a '
                        'validation dataset', default=10, type=int)
    parser.add_argument('--learning_rates', help='The learning rate or rates to use for each '
                        'tensorflow model', nargs='*', type=float, default=[1e-3])
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

    losses = []

    seen_pyod_models = set()

    Z_files = get_Z_files(args.Z_file_dir)

    for Z_file_path in Z_files:
        for seed in range(args.num_seeds):
            # For now, we'll use the load_data function from classifier.py.
            # In the future, we'll want to tinker with the Z df, so we'll implement a new one
            train_X, train_Y, val_X, val_Y = classifier.load_data(Z_file_path,
                                                                  args.healthy_file_path,
                                                                  args.disease_file_path,
                                                                  seed)
            val_baseline = utils.get_larger_class_percentage(val_Y)

            for lr in args.learning_rates:
                for model in model_list:
                    val_acc = None
                    val_auroc = None

                    if model_from_tf(model):
                        val_acc, val_auroc = eval_tf_model(model, lr, train_X, train_Y, val_X,
                                                           val_Y, args.checkpoint_dir, args.logdir,
                                                           args.epochs, seed)
                    # Don't rerun pyod models for different learning rates, because they don't have
                    # an lr hyperparameter
                    if (model, seed) not in seen_pyod_models:
                        if model_from_pyod(model):
                            val_acc, val_auroc = eval_pyod_model(model, train_X, train_Y,
                                                                 val_X, val_Y)
                            seen_pyod_models.add((model, seed))

                    losses.append((model, lr, seed, val_acc, val_auroc, val_baseline))

    results_df = pandas.DataFrame.from_records(losses, columns=['Model',
                                                                'LR',
                                                                'Seed',
                                                                'val_acc',
                                                                'val_auroc',
                                                                'val_baseline',
                                                                ])

    results_df.to_csv(args.out_path)
