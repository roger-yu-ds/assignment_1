

from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from pathlib import Path

project_dir = Path.cwd().parent
report_dir = project_dir / 'reports'
data_dir = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
interim_data_dir = data_dir / 'interim'
processed_data_dir = data_dir / 'processed'
models_dir = project_dir / 'models'

def authenticate(api):
    """
    Authenticates kaggle user
    """
    api.authenticate()
    print("Success! User is Authenticated")
    
    

def submit_predictions(classifier, X_test, test_id,message,pred_name='TARGET_5Yrs',pred_path=processed_data_dir/'preds_rf_cv.csv',competition = 'uts-advdsi-nba-career-prediction'):
    """
    Submits predictions to kaggle competition
    :param classifier: the classifier object
    :param X_test: the test dataset for submission
    :param test_id: test_id
    :param message: message associated with kaggle submission attempt
    
    :return:
    """
    load_dotenv(find_dotenv())
    api = KaggleApi()
    authenticate(api)
    preds = classifier.predict(X_test)
    submission = pd.DataFrame({'id':test_id,'TARGET_5Yrs': preds})
    submission.to_csv(pred_path, index = False)
    api.competition_submit(file_name=pred_path,
                       message=message,
                       competition=competition,
                       quiet=False)
    return submission