import logging
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from dotenv import find_dotenv, load_dotenv


def mnist():
    """Gets MNIST dataset from TensorFlow."""
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data(path="mnist.npz")

    return train_X, train_y, test_X, test_y


# creates command interface with two arguments.
@click.command()
# Where raw data will be housed.
@click.option("--raw_filepath", type=click.Path(exists=True))
# Where processed data will be housed.
@click.option("--processed_filepath", type=click.Path(exists=True))
def main(raw_filepath, processed_filepath):
    """Runs data processing scripts to convert raw data into normalized data."""
    logger = logging.getLogger(__name__)

    logger.info("Loading data.")
    train_X, train_y, test_X, test_y = mnist()

    logger.info("Checking data consistency.")
    assert train_X.shape == (60000, 28, 28)
    assert test_X.shape == (10000, 28, 28)
    assert train_y.shape == (60000,)
    assert test_y.shape == (10000,)

    logger.info(f"Saving raw data to {project_dir / raw_filepath}.")
    np.save(project_dir / raw_filepath / "train_X", train_X)
    np.save(project_dir / raw_filepath / "train_y", train_y)
    np.save(project_dir / raw_filepath / "test_X", test_X)
    np.save(project_dir / raw_filepath / "test_y", test_y)

    logger.info("Preprocessing data.")
    logger.info("Normalizing dataset.")
    train_X = train_X / 255.0
    test_X = test_X / 255.0

    logger.info(f"Saving 2D processed data to {project_dir / processed_filepath}.")
    np.save(project_dir / processed_filepath / "train_X_2D", train_X)
    np.save(project_dir / processed_filepath / "test_X_2D", test_X)

    logger.info("Flattening dataset.")
    train_X, test_X = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]), test_X.reshape(
        test_X.shape[0], test_X.shape[1] * test_X.shape[2]
    )

    logger.info(f"Saving 1D processed data to {project_dir / processed_filepath}.")
    np.save(project_dir / processed_filepath / "train_X_1D", train_X)
    np.save(project_dir / processed_filepath / "test_X_1D", test_X)

    logger.info(f"Saving labels to {project_dir / processed_filepath}.")
    np.save(project_dir / processed_filepath / "train_y", train_y)
    np.save(project_dir / processed_filepath / "test_y", test_y)

    logger.info("Done.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # working directory globally
    project_dir = Path(__file__).resolve().parents[2]

    # gets secret variables from .evn, which is atomatically added to .gitignore
    load_dotenv(find_dotenv())

    main()
