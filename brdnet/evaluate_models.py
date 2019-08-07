'''Compare the performance of the models in models.py on the validation dataset'''
import argparse
import importlib
import os

# Mute INFO and WARNINGs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas
from plotnine import *
import pyod
import sklearn

import classifier
import models
import utils

def model_from_pyod(model_name):
    try:
        # Import the model module from pyod
        model_module = 'pyod.models.{}'.format(model_name.lower())
        module = importlib.import_module(model_module)
        model = getattr(module, model_name)

        return True
    except (AttributeError, ModuleNotFoundError) as e:
        print(e)
        return False


def model_from_tf(model_name):
    try:
        model = getattr(models, model_name)
        return True
    except:
        return False


def eval_pyod_model(model_name, train_X, train_Y, val_X, val_Y, seed):
    # Import the model module from pyod
    model_module = 'pyod.models.{}'.format(model_name.lower())
    module = importlib.import_module(model_module)
    model = getattr(module, model_name)

    model_instance = model()
    model_instance.fit(train_X, train_Y)

    predictions = model_instance.predict(val_X)

    # Apply threshold_ on decision_scores_ from model after fit to get AUC

    val_acc = utils.calculate_accuracy(predictions, val_Y)
    val_auroc = sklearn.metrics.roc_auc_score(val_Y, model_instance.decision_function(val_X))

    return val_acc, val_auroc


def eval_tf_model(model, lr, train_X, train_Y, val_X, val_Y, checkpoint_dir, logdir, epochs, seed):
    lr_string = '{:.0e}'.format(lr)
    checkpoint_string = '{}_{}_{}'.format(model, lr_string, seed)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_string, 'checkpoint')

    os.makedirs(os.path.join(checkpoint_dir, checkpoint_string), exist_ok=True)

    classifier.train_model(train_X, train_Y, val_X, val_Y, checkpoint_path,
                           model_name=model, lr=lr,
                           epochs=int(epochs))

    # Load model
    untrained_model = utils.get_model(model, logdir, lr)

    val_loss, val_acc, val_auroc= untrained_model.evaluate(val_X, val_Y)

    untrained_model.load_weights(checkpoint_path)
    val_loss, val_acc, val_auroc = untrained_model.evaluate(val_X, val_Y)
    return val_acc, val_auroc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('Z_file_path', help='Path to the PLIER matrix to be used to convert '
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


    args = parser.parse_args()

    model_list = ['IForest', 'OCSVM']
    model_list.extend(utils.get_model_list())

    losses = []

    seen_models = set()

    for seed in range(1):
        args.seed = seed
        # For now, we'll use the load_data function from classifier.py.
        # In the future, we'll want to tinker with the Z df, so we'll implement a new one
        train_X, train_Y, val_X, val_Y = classifier.load_data(args)
        for lr in [10 ** -x for x in range(3,4)]:
            for model in model_list:
                val_acc = None
                val_auroc = None

                if model_from_tf(model):
                    val_acc, val_auroc = eval_tf_model(model, lr, train_X, train_Y, val_X, val_Y,
                                                       args.checkpoint_dir, args.logdir,
                                                       args.epochs, args.seed)
                # Don't rerun pyod models for different learning rates, because they don't have
                # an lr hyperparameter
                if (model, seed) not in seen_models:
                    if model_from_pyod(model):
                        val_acc, val_auroc = eval_pyod_model(model, train_X, train_Y, val_X, val_Y,
                                                             args.seed)
                        seen_models.add((model, seed))

                losses.append((model, lr, seed, val_acc, val_auroc))


    # Run eval on best model from each
    # Plot results

    loss_df = pandas.DataFrame.from_records(losses, columns=['Model',
                                                             'LR',
                                                             'Seed',
                                                             'val_acc',
                                                             'val_auroc',
                                                             ])
    loss_df.to_csv('loss_df.csv')

    plot = ggplot(loss_df, aes(x='Model', y='val_auroc', color='factor(LR)')) + geom_point()
    ggsave(plot, 'loss_plot.png')
    plot = ggplot(loss_df, aes(x='Model', y='val_acc', color='factor(LR)')) + geom_point()
    ggsave(plot, 'acc_plot.png')
