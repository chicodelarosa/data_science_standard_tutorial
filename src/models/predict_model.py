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
# Where predicted data will be housed.
@click.option('--predicted_filepath', type=click.Path(exists=True))
# Verbosity mode.
@click.option('--verbose', type=int, default=0)
def main(models_filepath, processed_filepath, predicted_filepath, verbose):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Loading model.')
    model = tf.keras.models.load_model(project_dir / models_filepath / 'baseline')

    logger.info('Loading training data.')
    test_X = np.load(project_dir / processed_filepath / 'test_X_2D.npy')
    test_y = np.load(project_dir / processed_filepath / 'test_y.npy')

    test_X_ = test_X.reshape((test_X.shape[0], 28, 28, 1))

    logger.info('Predicting test labels.')
    predicted_y_ = model.predict(test_X_, verbose = verbose)

    logger.info('Saving predicted labels.')
    np.save(project_dir / predicted_filepath / 'baseline', predicted_y_)

    logger.info('Done.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # working directory globally
    project_dir = Path(__file__).resolve().parents[2]

    # gets secret variables from .evn, which is atomatically added to .gitignore
    load_dotenv(find_dotenv())

    main()