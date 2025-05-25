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
    for col in df.columns:
        print("Col name: ", col)
        print("unique: ", df[col].nunique())
        print("null: ", df[col].isnull().sum(), "/", df[col].shape[0])
        print("-----------------------")
    # %%
    print("Target data: ", df[target_col].value_counts())


def show_distribution(df):
    for col in df.columns:
        plt.figure(figsize=(8, 6))

        if col == 'name':
            name_counts = df[col].value_counts()
            repeated_names = name_counts[name_counts > 1]
            plt.bar(repeated_names.index, repeated_names.values, color='purple')
            plt.title('Names (occurring more than once)')
            plt.xlabel('Name')
            plt.ylabel('Count')
            plt.xticks(rotation=45)

        elif col == 'gender' or col == 'work_type':
            gender_counts = df[col].value_counts(normalize=True) * 100
            plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', colors=['lightblue', 'lightcoral', 'lightgreen'])
            if col == 'gender':
                plt.title('Gender Distribution')
            elif col == 'work_type':
                plt.title('Work Type Distribution')
            plt.axis('equal')

        elif col == 'age':
            bins = range(0, int(df['age'].max()) + 10, 10)
            plt.hist(df['age'], bins=bins, color='skyblue', edgecolor='black')
            plt.title('Age Distribution')
            plt.xlabel('Age')
            plt.ylabel('Count')
            plt.xticks(bins)

        elif col == 'avg_glucose_level':
            plt.hist(df[col], bins=30, color='orange', edgecolor='black')
            plt.title('Average Glucose Level Distribution')
            plt.xlabel('Avg Glucose Level')
            plt.ylabel('Count')

        elif col == 'bmi':
            rounded_bmi = df[col].dropna().round().astype(int)
            counts = rounded_bmi.value_counts().sort_index()
            plt.bar(counts.index.astype(str), counts.values, color='green')
            plt.title('BMI Distribution (Rounded)')
            plt.xlabel('BMI')
            plt.ylabel('Count')

        else:
            counts = df[col].value_counts()
            plt.bar(counts.index.astype(str), counts.values, color='deeppink')
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
