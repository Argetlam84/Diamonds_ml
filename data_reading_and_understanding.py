import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def dataframe_reading(file_path):
    dataframe = pd.read_csv(file_path)
    return dataframe


def check_data(dataframe):
    print(20 * "-" + "Information".center(20) + 20 * "-")
    print(dataframe.info())
    print(20 * "-" + "Data Shape".center(20) + 20 * "-")
    print(dataframe.shape)
    print(20 * "-" + "Nunique".center(20) + 20 * "-")
    print(dataframe.nunique())
    print("\n" + 20 * "-" + "The First 5 Data".center(20) + 20 * "-")
    print(dataframe.head())
    print("\n" + 20 * "-" + "The Last 5 Data".center(20) + 20 * "-")
    print(dataframe.tail())
    print("\n" + 20 * "-" + "Missing Values".center(20) + 20 * "-")
    print((dataframe.isnull().sum()).sort_values(ascending=False))
    print("\n" + 40 * "-" + "Describe the Data".center(40) + 40 * "-")
    print(dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
  It returns categorical, numerical and cardinal variables names and quantity
  Parameters
  ----------
  dataframe: dataframe
      the dataframe where we take the variables names
  cat_th: int, float
      class threshold value for variables that are numeric but categorical
  car_th: int, float
      class threshold value for variables that are categorical but cardinal

  Returns
  -------
      cat_cols: list
          Categorical variables list
      num_cols: list
          Numerical variables list
      cat_but_car: list
          List of cardinal variables with categorical view
  Notes
  -----
  cat_cols + num_cols + cat_but_car = total variables
  num_but_cat in cat_cols

  """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    #cat_cols = [col for col in dataframe.columns if dataframe[col].dtype == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    #num_but_cat = [col for col in dataframe.columns if pd.Series(dataframe[col]).nunique() < cat_th]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05,0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90,0.95,0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################################################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.xticks(rotation=45)
        plt.show(block=True)


def target_summary(dataframe, target, column):
    if dataframe[column].dtypes == 'O':
        print(dataframe.groupby(target)[column].value_counts(), end="\n\n")

    else:
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print(dataframe.groupby(target).agg({column: 'mean'}), end="\n\n")
    print("###################################")


# def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
#     corr = dataframe.corr()
#     cor_matrix = corr.abs()
#     upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
#     drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
#     if plot:
#         import seaborn as sns
#         import matplotlib.pyplot as plt
#         sns.set(rc={'figure.figsize': (15, 15)})
#         sns.heatmap(corr, cmap="RdBu")
#         plt.show()
#     return drop_list

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu", vmin=-1, vmax=1)  # vmin ve vmax eklenerek renk aralığı sıkıştırılıyor
        plt.show()
    return drop_list

def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)

# def grab_col_names_plus(dataframe, cat_thr= 10,head=10, tail=10):
#    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
#
#    for col in cat_cols:
#        if dataframe[col].nunique() > cat_thr:
#            print("Col Name = ", dataframe[col], "Number Unique=", dataframe[col].nunique())
#            print(f"-------------------------------First {head} Observations----------------------------------")
#            print(dataframe[col].head(head))
#            print(f"-------------------------------Last {tail} Observations----------------------------------")
#            print(dataframe[col].tail(tail))
#            print(f"do you want to change {dataframe[col].dtype} if you want to input type")
#            input_type = input("Enter a type for example; date for datetime, int for integer, float for float")
#            if input_type == "date":
#                dataframe[col] = pd.to_datetime(dataframe[col])
#                return dataframe[col].dtype
#            elif input_type == "int":
#                dataframe[col] = dataframe[col].astype("int")
#                return dataframe[col].dtype
#            elif input_type == "float":
#                dataframe[col] = dataframe[col].astype("float")
#                return dataframe[col].dtype
#
#    return cat_cols, num_cols, cat_but_car


def grab_col_names_plus(dataframe, cat_thr=10, head=10, tail=10, input_type_dict=None):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    if input_type_dict is None:
        input_type_dict = {}

    for col in cat_cols:
        if dataframe[col].nunique() > cat_thr:
            print("Col Name =", col, "Number Unique =", dataframe[col].nunique())
            print(f"-------------------------------First {head} Observations----------------------------------")
            print(dataframe[col].head(head))
            print(f"-------------------------------Last {tail} Observations----------------------------------")
            print(dataframe[col].tail(tail))

            if col in input_type_dict:
                input_type = input_type_dict[col]
            else:
                input_type = input(f"Do you want to change {dataframe[col].dtype}? Enter a type (date, int, float, or leave empty): ")

            if input_type == "date":
                dataframe[col] = pd.to_datetime(dataframe[col])
            elif input_type == "int":
                dataframe[col] = dataframe[col].astype("int")
            elif input_type == "float":
                dataframe[col] = dataframe[col].astype("float")

            print(f"Column {col} has been converted to {dataframe[col].dtype}")
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    return cat_cols, num_cols, cat_but_car

def grab_col_names_and_change_num(dataframe, cat_thr=10, head=10, tail=10, input_type_dict=None):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    if input_type_dict is None:
        input_type_dict = {}

    for col in cat_but_car:
        if dataframe[col].nunique() > cat_thr:
            print("Col Name =", col, "Number Unique =", dataframe[col].nunique())
            print(f"-------------------------------First {head} Observations----------------------------------")
            print(dataframe[col].head(head))
            print(f"-------------------------------Last {tail} Observations----------------------------------")
            print(dataframe[col].tail(tail))

            if col in input_type_dict:
                input_type = input_type_dict[col]
            else:
                input_type = input(
                    f"Do you want to change {dataframe[col].dtype}? Enter a type (date, int, float, or leave empty): ")

            if input_type == "date":
                dataframe[col] = pd.to_datetime(dataframe[col])
            elif input_type == "int":
                dataframe[col] = dataframe[col].astype("int")
            elif input_type == "float":
                dataframe[col] = dataframe[col].astype("float")

            print(f"Column {col} has been converted to {dataframe[col].dtype}")
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    return cat_cols, num_cols, cat_but_car


def grab_col_names_and_change_cat(dataframe, cat_thr=10, head=10, tail=10, input_type_dict=None):
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    if input_type_dict is None:
        input_type_dict = {}

    for col in num_cols:
        if dataframe[col].nunique() < cat_thr:
            print("Col Name =", col, "Number Unique =", dataframe[col].nunique())
            print(f"-------------------------------First {head} Observations----------------------------------")
            print(dataframe[col].head(head))
            print(f"-------------------------------Last {tail} Observations----------------------------------")
            print(dataframe[col].tail(tail))

            if col in input_type_dict:
                input_type = input_type_dict[col]
            else:
                input_type = input(
                    f"Do you want to change {dataframe[col].dtype}? Enter a type (category, object, bool or leave empty): ")

            if input_type == "category":
                dataframe[col] = dataframe[col].astype('category')
            elif input_type == "object":
                dataframe[col] = dataframe[col].astype("object")
            elif input_type == "bool":
                dataframe[col] = dataframe[col].astype("bool")

            print(f"Column {col} has been converted to {dataframe[col].dtype}")
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    return cat_cols, num_cols, cat_but_car