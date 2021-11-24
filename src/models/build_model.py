# -*- coding: utf-8 -*-
import click
import logging
import tensorflow as tf
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


# creates command interface with one arguments.
@click.command()
# Where models will be housed.
@click.option('--models_filepath', type=click.Path(exists=True))
def main(models_filepath):
    """ Builds a 2D convolutional model and saves it to model_filepath.
    """
    logger = logging.getLogger(__name__)
    
    logger.info('Building 2D convolutional model.')
    
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    # compile model
    opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    logger.info(f'saving model to {project_dir / models_filepath}.')

    model.save(project_dir / models_filepath / "baseline")

    logger.info('Done.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # working directory globally
    project_dir = Path(__file__).resolve().parents[2]

    # gets secret variables from .evn, which is atomatically added to .gitignore
    load_dotenv(find_dotenv())

    main()