# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pickle
import tensorflow as tf
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# creates command interface with two arguments.
@click.command()
# Where model is stored.
@click.option('--models_filepath', type=click.Path(exists=True))
# Where processed data will be housed.
@click.option('--processed_filepath', type=click.Path(exists=True))
# Number of epochs to train the model for.
@click.option('--epochs', type=int, default=10)
# Training epoch batch size.
@click.option('--batch_size', type=int, default=32)
# Verbosity mode.
@click.option('--verbose', type=int, default=0)
def main(models_filepath, processed_filepath, epochs, batch_size, verbose):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Loading model.')
    model = tf.keras.models.load_model(project_dir / models_filepath / 'baseline')

    logger.info('Loading training data.')
    train_X = np.load(project_dir / processed_filepath / 'train_X_2D.npy')
    train_y = np.load(project_dir / processed_filepath / 'train_y.npy')

    train_X_ = train_X.reshape((train_X.shape[0], 28, 28, 1))
    train_y_ = tf.keras.utils.to_categorical(train_y)
    
    logger.info(f'Training model. Options: epochs={epochs}, batch_size={batch_size}, verbose={verbose}.')
    history = model.fit(train_X_, train_y_, epochs=epochs, batch_size=batch_size, verbose=verbose)
        
    logger.info('Saving trained model.')
    model.save(project_dir / models_filepath / 'baseline')

    logger.info('Saving training history.')
    with open(project_dir / processed_filepath / 'history.pkl', 'wb') as history_file:
        pickle.dump(history.history, history_file)

    logger.info('Done.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # working directory globally
    project_dir = Path(__file__).resolve().parents[2]

    # gets secret variables from .evn, which is atomatically added to .gitignore
    load_dotenv(find_dotenv())

    main()