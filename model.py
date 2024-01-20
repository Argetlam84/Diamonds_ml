import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
import variable_evaluations as ve


def evaluate_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = [('LR', LinearRegression()),
              # ("Ridge", Ridge()),
              # ("Lasso", Lasso()),
              # ("ElasticNet", ElasticNet()),
              ('KNN', KNeighborsRegressor()),
              ('CART', DecisionTreeRegressor()),
              ('RF', RandomForestRegressor()),
              # ('SVR', SVR()),
              ('GBM', GradientBoostingRegressor()),
              ("XGBoost", XGBRegressor(objective='reg:squarederror')),
              ("LightGBM", LGBMRegressor())]
    # ("CatBoost", CatBoostRegressor(verbose=False))]

    for name, regressor in models:
        rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
        xy = print(f"RMSE: {round(rmse, 4)} ({name}) ")
    return models[xy]


def evaluate_models_new(X, y, plot_imp=False, save=False, num=20):

    global fitted_models

    models = {
        'LR': LinearRegression(),
        'KNN': KNeighborsRegressor(),
        'CART': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Gradient Boosting': GradientBoostingRegressor(),
        'XGBoost': XGBRegressor(),
        'LightGBM': LGBMRegressor()
    }

    models_names = {}
    performance = {}

    for model_name, model in models.items():
        performance[model_name] = \
            {
               # "RMSE":  np.mean(np.sqrt(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")))

                "MAE": np.mean(np.abs(-cross_val_score(model, X, y, cv=5, scoring="neg_mean_absolute_error")))

            }
        models_names[model_name] = {model}

        fitted_models = [model.fit(X, y) for model_name, model in models.items()]

    if plot_imp:
        ve.plot_importance_for_func(fitted_models, X, num=num, save=save)

    return performance, models_names

from sklearn.metrics import mean_absolute_error