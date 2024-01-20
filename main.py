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

"""
Content
price price in US dollars ($326--$18,823)

carat weight of the diamond (0.2--5.01)

cut quality of the cut (Fair, Good, Very Good, Premium, Ideal)

color diamond colour, from J (worst) to D (best)

clarity a measurement of how clear the diamond is (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))

x length in mm (0--10.74)

y width in mm (0--58.9)

z depth in mm (0--31.8)

depth total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)
"""

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


