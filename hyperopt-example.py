import xgboost as xgb
from hyperopt import fmin, hp, tpe
from time import clock
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits
from sklearn.externals.joblib import load,dump
from sklearn.metrics import f1_score


def load_data():
    iris = load_digits(n_class=2)

    x = iris.data
    y = iris.target

    print(set(y))

    df = pd.DataFrame(x)
    df[4] = y

    sample = df.sample(150)

    sample = sample.reset_index(drop=True)

    dtrain = sample[0:70]

    dval = sample[70:120]

    dtest = sample[120:150]

    # print(sample)

    return dtrain, dval, dtest


def tune_xgb(dtrain,dval,dtest):


    dtrain = xgb.DMatrix(dtrain.ix[:,[1,2,3]], label=dtrain[4])
    dvalid = xgb.DMatrix(dval.ix[:,[1,2,3]], label=dval[4])
    dtest = xgb.DMatrix(dtest.ix[:,[1,2,3]], label=dtest[4])

    def objective(args):
        params = {
            'objective': 'binary:logistic',
            # 'num_class':3,
            'eval_metric': 'logloss',
            'nthread': 4,
            'eta': args['learning_rate'],
            'colsample_bytree': args['colsample_bytree'],
            'max_depth': args['max_depth'],
            'subsample': args['subsample'],
            'silent':1,
        }

        watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

        xgb1 = xgb.train(params, dtrain, 50, watchlist)

        pred_y = xgb1.predict(dtest)
        pred_y[pred_y >= 0.5] = 1
        pred_y[pred_y < 0.5] = 0
        test_y = dtest.get_label()

        f1 = f1_score(test_y,pred_y,average='binary')

        return f1


  # Searching space
    space = {
        'n_estimators': hp.quniform("n_estimators", 100, 200, 20),
        'learning_rate': hp.uniform("learning_rate", 0.01, 0.15),
        'max_depth': 8,
        'subsample': hp.uniform("subsample", 0.5, 0.9),
        'colsample_bytree': hp.uniform("colsample_bytree", 0.5, 0.9),
    }
    best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=150)
    print(best_sln)


if __name__ == '__main__':
    dtrain, dval, dtest = load_data()
    tune_xgb(dtrain, dval, dtest)
