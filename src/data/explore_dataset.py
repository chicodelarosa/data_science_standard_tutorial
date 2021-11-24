# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

# Creates command interface with two arguments.
@click.command()
# Where processed is housed.
@click.option('--processed_filepath', type=click.Path(exists=True))
# Where figures will be housed.
@click.option('--figures_filepath', type=click.Path(exists=True))
def main(processed_filepath, figures_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f'Loading data from {project_dir / processed_filepath}.')

    train_X = np.load(project_dir / processed_filepath / 'train_X_2D.npy')
    train_y = np.load(project_dir / processed_filepath / 'train_y.npy')

    logger.info('Plotting some samples.')

    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (10, 5))

    num2words = {1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five', 
                    6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 0: 'Zero'}

    extent = [0, 28, 0, 28]
    
    ax1.imshow(train_X[4701], cmap=plt.get_cmap('gray'), extent=extent)
    ax1.set_title(num2words[train_y[4701]])
    axes_range = range(0, 29, 7)

    ax1.set_xticks(axes_range)
    ax1.set_yticks(axes_range)

    ax2.imshow(train_X[4702], cmap=plt.get_cmap('gray'), extent=extent)
    ax2.set_title(num2words[train_y[4702]])
    ax2.set_xticks(axes_range)
    ax2.set_yticks(axes_range)
    
    ax3.imshow(train_X[4703], cmap=plt.get_cmap('gray'), extent=extent)
    ax3.set_title(num2words[train_y[4703]])
    ax3.set_xticks(axes_range)
    ax3.set_yticks(axes_range)

    fig.suptitle('MNIST Train Set Samples')

    fig.tight_layout()
    fig.savefig(project_dir / figures_filepath / 'training_samples.png')

    logger.info('Plotting class distribution.')
    
    fig, ax = plt.subplots()
    bins = range(11)
    ax.hist(train_y, bins=bins, color='crimson', linewidth=1, edgecolor='black')
    ax.set_xticks(range(10))
    ax.xaxis.set_minor_locator(tkr.AutoMinorLocator(n=2))
    ax.xaxis.set_minor_formatter(tkr.FixedFormatter(bins))
    ax.xaxis.set_major_formatter(tkr.NullFormatter())
    ax.set_xlabel('Class')
    ax.set_ylabel('Frequency')
    ax.set_title('MNIST Training Set Class Distribution')
    fig.savefig(project_dir / figures_filepath / 'class_ditribution.png')

    logger.info('Plotting train set class heatmap.')

    ncols = 5

    fig, axs = plt.subplots(nrows = 2, ncols = ncols, figsize = (15, 10))

    for row, col in zip([0] * 5 + [1] * 5, range(10)):
        axs[row][col % ncols].imshow(np.mean(train_X[train_y == col], axis=0), cmap=plt.get_cmap('coolwarm'), extent=extent)
        axs[row][col % ncols].set_title(num2words[col])

        axs[row][col % ncols].set_xticks(axes_range)
        axs[row][col % ncols].set_yticks(axes_range)
    
    fig.tight_layout()

    fig.suptitle('MNIST Train Set Class Heatmap')

    fig.savefig(project_dir / figures_filepath / 'trainset_class_heatmap.png')

    logger.info(f'Done.')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # working directory globally
    project_dir = Path(__file__).resolve().parents[2]

    # gets secret variables from .evn, which is atomatically added to .gitignore
    load_dotenv(find_dotenv())

    main()