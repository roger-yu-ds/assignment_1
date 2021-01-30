# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from typing import Tuple
import pandas as pd

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


def download_data(competition: str,
                  path,
                  unzip=True):
    """
    Downloads the required data set using the Kaggle api.
    Note that you need to first create a Kaggle API token, see https://www.kaggle.com/docs/api
    :param dataset: the name of the dataset
    :param path:
    :param unzip: object
    """
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition=competition,
                                   path=path)

    if unzip:
        for item in path.glob('*.zip'):
            with ZipFile(item, 'r') as zipObj:
                zipObj.extractall(path)


def separate_target(df: pd.DataFrame,
                    target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separates the target variable from the features.
    :param df: the dataframe that contains the full data
    :param target: the name of the target variable
    :return: a tuple of the features dataframe and the target series
    """
    X = df.copy(deep=True)
    y = df.pop(target)

    return X, y


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
