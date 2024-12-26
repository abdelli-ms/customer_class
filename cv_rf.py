from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import pandas as pd
import numpy as np
import math
import joblib

def sampled_range(mini, maxi, num):
    if not num:
        return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out
    
def transform_data_rf(df):
  """
  Apply transformations to the data.
  """
  df['Ever_Married'] = df['Ever_Married'] * (1 - df['Ever_Married_Unknown'])
  df['Work_Experience'] = df['Work_Experience'] * (1 - df['Work_Experience_Unknown'])
  df['Family_Size'] = df['Family_Size'] * (1 - df['Family_Size_Unknown'])
  df['Family_Size_2'] = df['Family_Size_2'] * (1 - df['Family_Size_Unknown'])
  df['Age_Exp'] = df['Age_Exp'] * (1 - df['Work_Experience_Unknown'])
  df['Graduated'] = df['Graduated'] * (1 - df['Graduated_Unknown'])
  return df

def preprocessing(data,kmeans):
  X_train = data.drop(columns="Segmentation")
  data = transform_data_rf(data)
  colstest = list(X_train.columns)
  labels = kmeans.predict(X_train)

  data['Cluster_Labels'] = labels

  data = pd.get_dummies(data, columns=['Cluster_Labels'], prefix='Cluster')

  cols =["Segmentation",
      "Segmentation_Encoded"]

  label_encoder = LabelEncoder()
  data["Segmentation_Encoded"] = label_encoder.fit_transform(data["Segmentation"])

  bins = [0, 1, 2, 6, 11, float('inf')]
  we_labels = [0, 1, 2, 3, 4]
  data['W_X_bin'] = pd.cut(data['Work_Experience'], bins=bins,labels=we_labels, include_lowest=True)
  data=data.drop(columns=["Work_Experience"])

  data["W_X_bin"]=data["W_X_bin"].astype(float)
  return data

def trainrf(cols_sel,rf_param_grid,kmeans):
  data_path = "./data/train.parquet"
  data = pd.read_parquet(data_path)
  data = preprocessing(data,kmeans)

  Y = data["Segmentation_Encoded"]
  X = data[cols_sel]

  print("train shape", X.shape)

  custom_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
  random_forest = RandomForestClassifier(random_state=0)
  grid_search = GridSearchCV(random_forest, rf_param_grid, cv=custom_cv, scoring='accuracy',n_jobs=8)
  grid_search.fit(X, Y)

  print("best parameters :", grid_search.best_params_)
  print("accuracy:", grid_search.best_score_)
  best_rf = grid_search.best_estimator_
  predictions = cross_val_predict(best_rf, X, Y, cv=custom_cv)
  accuracy = accuracy_score(Y, predictions)
  conf_matrix = confusion_matrix(Y, predictions)
  print("Confusion Matrix:\n", conf_matrix)
  print(f"\nAccuracy: {accuracy:.4f}")
  joblib.dump(best_rf, 'rf.pkl')
  
  return best_rf

rf_param_grid = {
    'n_estimators': [25,50,100, 200],       
    'max_depth': [2,4,8,10,16,20, 30, None],       
    'min_samples_split': [25,50,60,70,75,80,85,90,100,200],    
    'bootstrap': [True, False]               
}

cols_sel=['Age_Unknown', 'Artist', 'Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7', 'Doctor', 'Engineer', 'Entertainment', 
  'Ever_Married', 'Ever_Married_Unknown', 'Executive', 'Family_Size', 'Family_Size_Unknown',
  'Gender', 'Graduated', 'Graduated_Unknown', 'Healthcare', 'Homemaker', 'Lawyer', 'Log_Age', 'Log_Spending_Score_Age',
  'Marketing','Profession_Unknown', 'Spending_Score', 'Spending_Score_Age', 'Work_Experience_Unknown',
  'Cluster_0', 'Cluster_1','Cluster_2', 'Cluster_3',"Age",'W_X_bin']

kmeans = joblib.load('kmeans_model.pkl')
best_rf=trainrf(cols_sel=cols_sel,rf_param_grid=rf_param_grid,kmeans=kmeans)

test_data_path = "./data/test.parquet"
test_data = pd.read_parquet(test_data_path)

test_data = preprocessing(test_data,kmeans)
X_test=test_data[cols_sel]
Y_test = test_data["Segmentation_Encoded"]
print("test shape", X_test.shape)
test_predictions = best_rf.predict(X_test)
test_conf_matrix = confusion_matrix(Y_test, test_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)
print("-"*40)
print("Test Confusion Matrix:\n", test_conf_matrix)
print(f"Test Accuracy: {test_accuracy:.4f}")