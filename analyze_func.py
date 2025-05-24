import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def simple_analysis(df, target_col='stroke'):
    """
    Function to analyze
    :param df:
    :return:
    """

    print("All columns: ", df.columns)
    # %%
    print("Shape of df: ", df.shape)
    # %%
    print("Description of df:\n", df.describe())
    # %%

    # %% md
    # Za dużo trzeba zmiejszyć no stroke lub stucznie dotwoarzyć, jednak przez duża liczbę dancyh, usuwamy
    # %%
    for col in df.columns:
        print("Col name: ", col)
        print("unique: ", df[col].nunique())
        print("null: ", df[col].isnull().sum(), "/", df[col].shape[0])
        print("-----------------------")
    # %%
    print("Target data: ", df[target_col].value_counts())


def show_distribution(df):
    for col in df.columns:
        quality_counts = df[col].value_counts()

        plt.figure(figsize=(8, 6))
        plt.bar(quality_counts.index.astype(str), quality_counts, color='deeppink')
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def col_unique(df, cols=None):
    if cols is None:
        cols = ['Gender', 'Marital Status', 'Work Type', 'Residence Type', 'Smoking Status', 'Alcohol Intake',
                'Physical Activity', 'Family History of Stroke', 'Dietary Habits']

    for col in cols:
        print("Name: ", col, "\nUnique Values: ", df[col].unique(), "\n")


def show_all_symptoms(df, col='Symptoms'):
    all_comb_symp = []
    for el in df[col]:
        el = str(el)
        el_list = el.split(', ')
        for el2 in el_list:
            # if el2[0] == " ":
            #     el2 = el2[:1]
            if el2 not in all_comb_symp:
                all_comb_symp.append(el2)

    all_comb_symp.remove('nan')
    print("All symptoms: ", all_comb_symp)

    return all_comb_symp

#
# def prepare_gender(row, ):
#
#     if row == 'Female':
#         return 1.0
#     else:
#         return 0.0

def unify_values(row, sympt):

    if row == sympt:
        return 1
    else:
        return 0

def split_BPL(row):
    row = str(row)
    row = row.split('/')
    return row[0], row[1]

def split_Chrolesterol(row):
    row = str(row)
    row = row.split(',')
    return row[0][5:], row[1][5:]

def change_sympt(row, sympt):
    row = str(row)
    if row != "nan":
        if sympt in row:
            return 1
        else:
            return 0
    else:
        return 0
