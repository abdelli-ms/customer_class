from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def describing(df):
  """
  prints describe in a plausible format ( four columns at a time)
  input : dataframe
  returns Nothing, 
  """
  for idx in range(0,len(df.columns),4):
    end_idx = min(idx + 4, len(df.columns)) 
    print(df.iloc[:, idx:end_idx].describe())
  print("------------------------------")
  print(df.dtypes)
  print("------------------------------")
def random_forest(X,Y):
  """
  X is training data
  Y is lables of the training set
  this matrix prints 17 feature of the most important features of the dataset
  """
  rf = RandomForestClassifier(random_state=0)
  rf.fit(X, Y)
  importances = rf.feature_importances_
  feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
  top_features = feature_importance.head(17)['Feature'].tolist()
  print(feature_importance)
  print(top_features)

def plot_correlation_heatmap(correlation_matrix):
  """
  this function prints a heatmap of a matrix 
  in my case I use it for correlation heatmap
  input:
  correlation heatmap
  output:
  prints a figure ( heatmap )
  """
  plt.figure(figsize=(10, 8))
  sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
  plt.title("Correlation Heatmap")
  plt.show()
def infer(df,min,std):
  df['Age_2'] = df['Age'] ** 2
  df['Log_Age'] = np.log(df['Age'])
  df['Age_Normalized']=(df["Age"]-min)/std
  df['Log_Age_Normalized']=np.log(df['Age_Normalized']+1)

  df['Spending_Score_Age'] = (df['Spending_Score'] +1)* data['Age']
  df['Log_Spending_Score_Age'] = np.log(df["Spending_Score_Age"])
  df["Spending_Score_Age_2"]=df["Spending_Score_Age"]**2

  df['Family_Size_2'] = df['Family_Size'] ** 2
  df['Log_Family_Size'] = np.log(df['Family_Size']+1)

  df["Work_Experience_2"]=df["Work_Experience"]**2
  df["Log_Work_Experience"]=np.log(df["Work_Experience"]+1)

  df['Age_Exp'] = df.apply(lambda row: row['Work_Experience'] / row['Age'] if row['Age'] != 0 else 0, axis=1)
  df['Age_Exp_2'] = df['Age_Exp'] ** 2
  df["Log_Age_Exp"]=np.log(df["Age_Exp"]+1)
  df['Age_Exp'] = df.apply(lambda row: row['Work_Experience'] / row['Age'] if row['Age'] != 0 else 0, axis=1)
  return df

train_file='./data/train_data.parquet'
test_file='./data/test_data.parquet'
data_t = pd.read_parquet(test_file)
data = pd.read_parquet(train_file)

age_min = data["Age"].min()
age_std=data["Age"].std()

data = infer(data,age_min,age_std)

data_t = infer(data_t,age_min,age_std)



Y = data["Segmentation"]
X = data.drop(columns=["Segmentation"])
Y_t = data_t["Segmentation"]
X_t = data_t.drop(columns=["Segmentation"])

# data exploration section: correlation and random forest
# random_forest(X,Y)
# correlations = X.corr()
# plot_correlation_heatmap(correlations)
data_path="./data/train.parquet"
data_t_path="./data/test.parquet"
data.to_parquet(data_path, engine='pyarrow', index=False)
data_t.to_parquet(data_t_path, engine='pyarrow', index=False)
