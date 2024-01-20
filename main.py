import data_reading_and_understanding as dr
import feature_engineering as fe
import variable_evaluations as ve
import model as ml
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 500)
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

df_d = sns.load_dataset("diamonds")

dr.check_data(df_d)

cat_cols, num_cols, cat_but_car = dr.grab_col_names(df_d)

dr.num_summary(df_d, num_cols, plot=True)

for col in cat_cols:
    dr.cat_summary(df_d, col, plot=True)


dr.high_correlated_cols(df_d[num_cols], plot=True)


dr.correlation_matrix(df_d,num_cols)

df_d = fe.one_hot_encoder(df_d,cat_cols)

X = df_d.drop("price", axis=1)
y = df_d["price"]

performance, models = ml.evaluate_models_new(X,y,plot_imp=True)


