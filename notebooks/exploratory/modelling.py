#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from scipy import stats
from joblib import dump
from joblib import load
import xgboost as xgb
import matplotlib.pyplot as plt
from typing import Dict
from src.data import make_dataset
from kaggle.api.kaggle_api_extended import KaggleApi
from dotenv import find_dotenv, load_dotenv


# In[78]:


load_dotenv(find_dotenv())
api = KaggleApi()
api.authenticate()


# In[80]:


competition = os.environ['COMPETITION']


# # Set up directories

# In[65]:


project_dir = Path.cwd().parent
data_dir = project_dir / 'data'
raw_data_dir = data_dir / 'raw'
interim_data_dir = data_dir / 'interim'
processed_data_dir = data_dir / 'processed'
models_dir = project_dir / 'models'


# # Load data

# In[57]:


df_train = pd.read_csv(raw_data_dir / 'train.csv')
df_test = pd.read_csv(raw_data_dir / 'test.csv')
X_train = np.load(interim_data_dir / 'X_train.npy')
X_val = np.load(interim_data_dir / 'X_val.npy')
y_train = np.load(interim_data_dir / 'y_train.npy')
y_val = np.load(interim_data_dir / 'y_val.npy')
X_test = np.load(interim_data_dir / 'X_test.npy')
test_id = pd.read_csv(interim_data_dir / 'test_id.csv')


# # Baseline
# 
# The base line prediction is simply to make them all negative.

# In[36]:


labels = 'Positive', 'Negative'
pos_count = (y_train == 1).sum()
neg_count = (y_train == 0).sum()
sizes = [pos_count, neg_count]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[37]:


labels = 'Positive', 'Negative'
pos_count = (y_val == 1).sum()
neg_count = (y_val == 0).sum()
sizes = [pos_count, neg_count]
explode = (0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.show()


# In[38]:


preds = [1] * len(y_val)
roc_auc_score(y_val, preds)


# # XGB

# In[39]:


clf_xgb = xgb.XGBClassifier()


# In[40]:


clf_xgb.fit(X_train, y_train)


# In[41]:


preds = clf_xgb.predict(X_val)
probs = clf_xgb.predict_proba(X_val)


# In[42]:


X_val.shape


# In[43]:


len(y_val)


# In[44]:


auc = roc_auc_score(y_val, probs[:, 1])
tpr, fpr, threshold = roc_curve(y_val, probs[:, 1])
auc


# In[ ]:





# # RandomizedSearchCV

# In[45]:


# test
df_train.info()


# In[46]:


pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA()),
        ('classifier', xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        ))
])

param_dist = {
    'pca__n_components': stats.randint(1, X_train.shape[1]),
    'classifier__n_estimators': stats.randint(150, 1000),
    'classifier__learning_rate': stats.uniform(0.01, 0.6),
    'classifier__subsample': stats.uniform(0.3, 0.9),
    'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9],
    'classifier__colsample_bytree': stats.uniform(0.5, 0.9),
    'classifier__min_child_weight': [1, 2, 3, 4]
}

cv = RandomizedSearchCV(
    estimator=pipe,
    param_distributions=param_dist,
    random_state=42,
    n_iter=100,
    cv=5,
    n_jobs=7,
    verbose=10
)

cv.fit(X_train, y_train)


# In[60]:


dump(cv, models_dir / 'randomised_xgb')


# In[47]:


pd.DataFrame(cv.cv_results_)


# ## Predictions with the best model

# In[48]:


preds = cv.predict(X_val)
probs = cv.predict_proba(X_val)


# In[49]:


len(probs[:, 1])


# In[50]:


fpr, tpr, thresholds = roc_curve(y_val, probs[:, 1])
roc_auc_score(y_val, probs[:, 1])


# In[51]:


confusion_matrix(y_val, preds)


# ## Predict on test set

# In[68]:


preds = cv.predict(X_test)


# ## Save predictions

# In[81]:


pred_name = 'TARGET_5Yrs'
pred_path = processed_data_dir / 'preds_randomised_xgb.csv'
make_dataset.save_predictions(preds, pred_name, test_id, pred_path)


# In[83]:


pred_path.stem


# ## Submit predictions

# In[85]:


api.competition_submit(file_name=pred_path,
                       message=pred_path.stem,
                       competition=competition,
                       quiet=False)


# ## Predictions 2nd best model

# There's a very small difference in the `mean_test_score`s of the first and second. The second ranked model uses only 2 `pca_n_components`. The best model on the training set might be overfitting due to the large number of components. Let's try the second best model on the validation set.

# In[53]:


type(cv)


# In[61]:


def get_parameters(cv: RandomizedSearchCV,
                   step: str,
                   nth: int) -> Dict[str, float]:
    """
    Extract the parameters of the non-first ranked model from a RandomizedSearchCV object,
    so that they may be used to fit another model.
    """
    key_list = [key for key in cv.cv_results_.keys() if step in key]
    return key_list

