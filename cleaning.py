#imports and libraries
import pandas as pd
import sys
import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
#global variables
data_folder ="data"
test_csv_name = "or_test.csv"
train_csv_name = "or_train.csv"
list_files = [test_csv_name,train_csv_name]
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
    df : adataframe
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
def to_list_cats(df):
  """
  a debugging function, takes a dataframe and prints the values of the categories for each category column
  """
  print("------------------------------")
  for column in df.columns:
    if(df[column].dtype.name=="category"):
      print(column,df[column].cat.categories)
  print("------------------------------")

  
list_cats=["Gender","Ever_Married","Graduated","Profession","Spending_Score","Var_1","Segmentation"]
cat_dtypes.update(columns_dtypes)
all_data = read_all_data(data_folder,list_files,cat_dtypes)
all_data=handle_None_cat(all_data,list_cats)
all_data = all_data.drop_duplicates(keep="first")
list_todummies = ["Gender","Ever_Married","Graduated","Profession","Var_1","Segmentation"]
all_data = to_dummies(all_data,list_todummies)
all_data["Spending_Score"]=all_data["Spending_Score"].map({'Low': 1.0, 'Average': 2.0, 'High': 3.0}).astype(float)
cols_to_drop =["Ever_Married_No","Graduated_No","Gender_Female","Gender","Ever_Married","Graduated","Profession","Var_1","Segmentation"]
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
cols_nulls = ["Work_Experience", "Family_Size"]# these columns are the only ones containing nulls now
print("|========================================================================|")
all_data['Work_Experience_Missing'] = all_data['Work_Experience'].notna().astype(float)
all_data['Work_Experience'] = all_data['Work_Experience'].fillna(all_data['Work_Experience'].mean())
all_data.reset_index(drop=True, inplace=True)
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)


# Create the histogram
plt.hist(all_data['Graduated'], bins=30, edgecolor='black')

# Add title and labels
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')

# Show the plot
plt.show()
plt.close()