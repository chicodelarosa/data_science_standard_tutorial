# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report

def mnist():
    """
        Gets MNIST dataset from TensorFlow.
    """
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(path = 'mnist.npz')

    return train_X, train_y, test_X, test_y


# creates command interface with two arguments.
@click.command()
# Where processed data is housed.
@click.option('--processed_filepath', type=click.Path(exists=True))
# Where predicted labels are housed.
@click.option('--predicted_filepath', type=click.Path(exists=True))
# Where figures will be housed.
@click.option('--figures_filepath', type=click.Path(exists=True))
# Where textual reports will be housed.
@click.option('--scores_filepath', type=click.Path(exists=True))
def main(processed_filepath, predicted_filepath, figures_filepath, scores_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Loading ground truth label data.')
    test_y = np.load(project_dir / processed_filepath / 'test_y.npy')

    logger.info('Loading predicted label data.')
    predicted_y = np.load(project_dir / predicted_filepath / 'baseline.npy')
    predicted_y_ = np.argmax(predicted_y, axis = 1)

    logger.info('Computing confusion matrix.')
    cm = confusion_matrix(test_y, predicted_y_)

    logger.info('Plotting confusion matrix.')
    sns.heatmap(cm.astype(int), linewidth=0.5, cmap='YlGnBu', square=True, annot=True, cbar_kws={'label': 'Count'}, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')

    logger.info('Saving confusion matrix plot.')
    plt.savefig(project_dir / figures_filepath / 'confusion_matrix.png')

    logger.info('Computing classification scores.')
    cr = classification_report(test_y, predicted_y_)

    logger.info(f"Saving classification scores to {project_dir / scores_filepath / 'classification_report.txt'}.")
    with open(project_dir / scores_filepath / 'classification_report.txt', 'w') as class_report:
        class_report.write(cr)

    logger.info(f'Done.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # working directory globally
    project_dir = Path(__file__).resolve().parents[2]

    # gets secret variables from .evn, which is atomatically added to .gitignore
    load_dotenv(find_dotenv())

    main()