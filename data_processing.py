import settings
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
np.random.seed(1)

def load_dataset():
    df_train = pd.read_csv(settings.TRAIN_RAW_PATH).set_index('Id')
    df_train['dataset'] = 'train'
    df_test = pd.read_csv(settings.TEST_RAW_PATH).set_index('Id')
    df_test['dataset'] = 'test'
    df = pd.concat((df_train, df_test))
    return df


def preprocess_dataset(df, le_type=None, replace_nan=True):
    le_d = {}
    if le_type == 'label':
        for col in df.columns:
            if df[col].dtype == object and col != 'dataset':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                le_d[col] = le
    if replace_nan:
        df = df.replace(np.nan, 0)
    return df, le_d


def split_data(df):
    df_train = df[df['dataset'] == 'train'].copy()
    df_train.drop(columns='dataset', inplace=True)
    df_test = df[df['dataset'] == 'test'].copy()
    df_test.drop(columns=['dataset', 'SalePrice'], inplace=True)
    return df_train, df_test


def train_model(mdl, param, df):
    df_train, df_test = split_data(df)
    clf = GridSearchCV(mdl, param, verbose=3, scoring='neg_root_mean_squared_error', n_jobs=-1)
    clf.fit(df_train.drop(columns='SalePrice'), np.log(df_train['SalePrice']))
    print(np.abs(clf.best_score_))
    df_test['SalePrice'] = np.exp(clf.predict(df_test))
    return clf, df_train, df_test


def save_submission(df_test):
    df_test['SalePrice'].to_csv('C:/Users/ariav/Desktop/out.csv')
