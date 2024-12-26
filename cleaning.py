#imports and libraries
import pandas as pd
import sys
import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split

def read_all_data(relative_ddir_name,list_dir_names,type_dict):
  """
  this function reads data from multiple files into a single dataframe
  input:
    relative_ddir_name: str relative path to data directory
    list_dir_names : list of str each correxponding to a csv file containing data
  output:
    pandas dataframe with ID column dropped and index reset and duplicates dropped
  """
  working_dir = os.getcwd()
  data_dir = os.path.join(working_dir,relative_ddir_name)
  file_paths=[]
  if(not os.path.exists(data_dir)):
    print("wrong path for data dir")
    return None
  for file in list_dir_names:
    if(file!=None):
      file_dir = os.path.join(data_dir,file)
      if(os.path.exists(file_dir)):
        file_paths.append(file_dir)
  dataframes = []
  for file in file_paths:
    dataframes.append(pd.read_csv(file,sep=',',dtype=type_dict))
  df = pd.concat(dataframes)
  df.drop(columns=["ID"],inplace=True)
  df.reset_index(drop=True,inplace=True)
  return df

def handle_None_cat(dataf,columns,val_cat="Unknown"):
  """
  the purpose of this function is to take a list of categorical columns and convert missing 
  values into val_cat category for each of the columns (we will see later if it is okay) 
  input:
    dataf: a pandas Dataframe
    columns: a list of categorical columns
    val_cat: value of the category that we want to assign for missing values
  output:
    dataframe with missing values replaced by val_cat for columns in list columns
  """
  for column in columns:
    if(column in dataf.columns and dataf[column].isnull().any()):
      if(dataf[column].dtype.name!="category"):
        print("column <",column,"> is not of type category")
        continue
      else:
        dataf[column]=dataf[column].cat.add_categories(val_cat).fillna(val_cat)
    else:
      print("<",column,">: is  not a column of the dataframe or has no null values")
  return dataf
def to_dummies(df,list_cols):
  """
  takes a dataframe and creates dummies for list of columns that are categorical.
  input: 
    df : a dataframe
    list_cols : a list of columns
  output : 
    a dataframe for which each of the categorical column in the list list_cols is converted into dummy columns
  """
  for col in list_cols:
    if(col not in df.columns):
      continue
    if(df[col].dtype.name=="category"):
      df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
      df_dummies=df_dummies.astype(float)
      df = pd.concat([df, df_dummies], axis=1)
  return df

def to_list_cats(df):
  """
  a debugging function, takes a dataframe and prints the values of the categories for each category column
  """
  print("------------------------------")
  for column in df.columns:
    if(df[column].dtype.name=="category"):
      print(column,df[column].cat.categories)
  print("------------------------------")
def num_normalization(dataf,cols):
  """
  a function to clean numerical columns
  input:
  dataf: dataframe
  cols: the columns to normalize
  output:
  return the dataframe with filtered data
  """
  for col in cols:
    if(dataf[col].dtype.name!="float64"):
      print(f"<{col}> is not of type flaot")
      continue
    if(not dataf[col].isna().any()):
      dataf[f"{col}_Unknown"]=0.0
    else:
      dataf[f"{col}_Unknown"]=dataf[col].isna().astype(float)
    #dataf[col] = (dataf[col] - dataf[col].min()) / dataf[col].std()
    dataf[col] = dataf[col].fillna(dataf[col].mean())
  return dataf

#global variables
data_folder ="data"
test_csv_name = "or_test.csv"
train_csv_name = "or_train.csv"
list_files = [test_csv_name,train_csv_name]
cat_dtypes = defaultdict(lambda : "object")
columns_dtypes = {
  "ID":"object",
  "Gender":"category",
  "Ever_Married":"category",
  "Age": "float64",
  "Graduated":"category",
  "Profession":"category",
  "Work_Experience":"float64",
  "Spending_Score":"category",
  "Family_Size":"float64",
  "Var_1":"category",
  "Segmentation":"category",
}
list_cats=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1","Segmentation"]
cat_dtypes.update(columns_dtypes)
all_data = read_all_data(data_folder,list_files,cat_dtypes)
all_data=handle_None_cat(all_data,list_cats)
list_todummies = ["Gender","Ever_Married","Graduated","Profession","Var_1"]
all_data = to_dummies(all_data,list_todummies)
all_data["Spending_Score"]=all_data["Spending_Score"].map({'Low': 1.0, 'Average': 2.0, 'High': 3.0}).astype(float)
cols_to_drop =["Ever_Married_No","Graduated_No","Gender_Female","Gender","Ever_Married","Graduated","Profession","Var_1"]
all_data=all_data.drop(columns=cols_to_drop)
rename_dict ={
      'Gender_Male':'Gender', 
      'Ever_Married_Yes':'Ever_Married',
       'Graduated_Yes' : 'Graduated', 
       'Profession_Artist':'Artist',
       'Profession_Doctor':'Doctor', 
       'Profession_Engineer':'Engineer', 
       'Profession_Entertainment':'Entertainment',
       'Profession_Executive':'Executive',
       'Profession_Healthcare':'Healthcare', 
       'Profession_Homemaker':'Homemaker',
       'Profession_Lawyer':'Lawyer', 
       'Profession_Marketing':'Marketing', 
       'Var_1_Cat_1':'Cat_1',
       'Var_1_Cat_2':'Cat_2', 
       'Var_1_Cat_3':'Cat_3', 
       'Var_1_Cat_4':'Cat_4',
       'Var_1_Cat_5':'Cat_5', 
       'Var_1_Cat_6':'Cat_6', 
       'Var_1_Cat_7':'Cat_7',
}
all_data=all_data.rename(columns=rename_dict)
cols_nums = ["Work_Experience", "Age","Family_Size"]
all_data = num_normalization(all_data,cols_nums)
all_data = all_data.drop_duplicates(keep="first")
all_data.reset_index(drop=True, inplace=True)
train_data, test_data = train_test_split(all_data, test_size=0.3, random_state=42,stratify=all_data["Segmentation"])
test_file ='./data/test_data.parquet'
train_file='./data/train_data.parquet'
test_data.to_parquet(test_file, engine='pyarrow', index=False)
train_data.to_parquet(train_file, engine='pyarrow', index=False)
