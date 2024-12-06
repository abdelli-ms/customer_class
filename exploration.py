from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def describeing(df):
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
  rf = RandomForestClassifier(random_state=0)
  rf.fit(X, Y)
  importances = rf.feature_importances_
  feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
  top_features = feature_importance.head(17)['Feature'].tolist()
  print(feature_importance)
  print(top_features)

def plot_correlation_heatmap(correlation_matrix):
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()
def store_data(dest_path,df):
  pass

train_file='./data/train_data.parquet'
data = pd.read_parquet(train_file)
Y = data["Segmentation"]
X = data.drop(columns=["Segmentation"])

X['Age_Exp'] = X.apply(lambda row: row['Work_Experience'] / row['Age'] if row['Age'] != 0 else 0, axis=1)
data['Age_Exp']=X['Age_Exp']
X['Age_2'] = X['Age'] ** 2
data['Age_2'] = X['Age_2']
X['Spending_Score_Age'] = X['Spending_Score'] * X['Age']
data['Spending_Score_Age']=X['Spending_Score_Age']
X['Family_Size_2'] = X['Family_Size'] ** 2
data['Family_Size_2']=X['Family_Size_2']
X['Work_Exp_Fam_Ratio'] = X['Work_Experience'] / (X['Family_Size'] + 1e-6)
data['Work_Exp_Fam_Ratio']=X['Work_Exp_Fam_Ratio']
X['Log_Age'] = np.log(X['Age'] + 1)
data['Log_Age'] = X['Log_Age']
X['Age_Exp_2'] = X['Age_Exp'] ** 2
data['Age_Exp_2']=X['Age_Exp_2']

# display statistics: 
# random_forest(X,Y)
# correlations = X.corr()
# plot_correlation_heatmap(correlations)
# print(correlations)
# print(data.columns)
data_path="./data/data_extra.parquet"
data.to_parquet(data_path, engine='pyarrow', index=False)
