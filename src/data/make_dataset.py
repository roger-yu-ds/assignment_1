# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from kaggle.api.kaggle_api_extended import KaggleApi
from zipfile import ZipFile
from typing import Tuple, List
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
    y = X.pop(target)

    return X, y


def make_adversarial_validation_dataset(df_train: pd.DataFrame,
                                        df_test: pd.DataFrame,
                                        cols: List[str],
                                        n: int = None):
    """
    Creates a training and testing sets for adversarial validation
    :param df_train: the original training dataframe
    :param df_test: the original test dataframe
    :param cols: the columns to be included in the adversarial data sets
    :param n: the size of the adversarial data set, defaults to the size of the
    original data
    :return: a tuple of the adversarial training and test sets
    """
    if cols is None:
        cols = df_train.columns

    df = pd.concat([df_train[cols].assign(dataset='train'),
                    df_test[cols].assign(dataset='test')])

    # The number of test samples is the number of samples in the original
    # test set
    if n is None:
        n = df_train.shape[1]

    adv_train = df.sample(n=n, replace=False)
    adv_test = df.loc[~df.index.isin(adv_train.index)]

    return adv_train, adv_test

def save_predictions(preds, pred_name, id_df, path):
    """
    Save the predictions of the model on to disk
    :param preds:
    :param id_df:
    :param path:
    :return:
    """
    pred_df = id_df.copy(deep=True)
    pred_df.loc[:, pred_name] = preds
    pred_df.to_csv(path, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
