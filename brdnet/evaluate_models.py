'''Compare the performance of the models in models.py on the validation dataset'''
import argparse
import os

# Mute INFO and WARNINGs from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas
from plotnine import *

import classifier
import utils



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
    parser.add_argument('--logdir', help='The directory to log training progress to',
                        default='../logs')


    args = parser.parse_args()

    model_list = utils.get_model_list()

    losses = []

    for seed in range(20):
        args.seed = seed
        # For now, we'll use the load_data function from classifier.py.
        # In the future, we'll want to tinker with the Z df, so we'll implement a new one
        train_X, train_Y, val_X, val_Y = classifier.load_data(args)
        for lr in [10 ** -x for x in range(3,5)]:
            for model in model_list:
                lr_string = '{:.0e}'.format(lr)
                checkpoint_string = '{}_{}_{}'.format(model, lr_string, seed)
                checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_string, 'checkpoint')

                os.makedirs(os.path.join(args.checkpoint_dir, checkpoint_string), exist_ok=True)

                classifier.train_model(train_X, train_Y, val_X, val_Y, checkpoint_path,
                                       model_name=model, logdir=args.logdir, lr=lr,
                                       epochs=int(args.epochs))

                # Load model
                untrained_model = utils.get_model(model, args.logdir, lr)

                val_loss, val_acc = untrained_model.evaluate(val_X, val_Y)

                untrained_model.load_weights(checkpoint_path)
                val_loss, val_acc = untrained_model.evaluate(val_X, val_Y)

                losses.append((model, lr, seed, val_loss, val_acc))


    # Run eval on best model from each
    # Plot results

    loss_df = pandas.DataFrame.from_records(losses, columns=['Model',
                                                             'LR',
                                                             'Seed',
                                                             'val_loss',
                                                             'val_acc',
                                                             ])

    print(loss_df)

    plot = ggplot(loss_df, aes(x='Model', y='val_loss', color='factor(LR)')) + geom_point()
    ggsave(plot, 'loss_plot.png')
    plot = ggplot(loss_df, aes(x='Model', y='val_acc', color='factor(LR)')) + geom_point()
    ggsave(plot, 'acc_plot.png')
