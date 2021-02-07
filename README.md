assignment_1
==============================

MDSI 36114 ADSI Assignment 1



Data Dictionary
[Kaggle link](https://www.kaggle.com/c/uts-advdsi-nba-career-prediction/data?select=sample_submission.csv)
The name of the data is `uts-advdsi-nba-career-prediction`.
----------------------------
| column       | description                                         |
|--------------|-----------------------------------------------------|
| Id           | Player Identifier                                   |
| GP           | Games Played                                        |
| MIN          | Minutes Played                                      |
| PTS          | Points Per Game                                     |
| FGM          | Field Goals Made                                    |
| FGA          | Field Goals Attempts                                |
| FG%          | Field Goals Percent                                 |
| 3P Made      | 3 Points Made                                       |
| 3PA          | 3 Points Attempts                                   |
| 3P%          | 3 Points Percent                                    |
| FTM          | Free Throw Made                                     |
| FTA          | Free Throw Attempts                                 |
| FT%          | Free Throw Percent                                  |
| OREB         | Offensive Rebounds                                  |
| DREB         | Defensive Rebounds                                  |
| REB          | Rebounds                                            |
| AST          | Assists                                             |
| STL          | Steals                                              |
| BLK          | Blocks                                              |
| TOV          | Turnovers                                           |
| TARGET_5Yrs  | Outcome: 1 if career length >= 5 years, 0 otherwise |

Prerequisites
------------
* Python 3
* Pip 
* a kaggle account


Getting Started
------------
It is suggested that before starting this project, you create a virtual environment. By installing the `pipenv` package, this will be done automatically according to the pipfile. Then activate the environment accordingly:
```bash
pip install pipenv
pipenv shell
```

To install required packages:

```bash
 pipenv install -r requirements.txt
```

Ensure that you have retrieved a kaggle API key and store the resulting JSON file in `/user/.kaggle/kaggle.json`

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

Kaggle API
--------------------------
For documentation on how to interact with the Kaggle API within Python, go to https://technowhisp.com/kaggle-api-python-documentation/.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
