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
  df['Work_Exp_Fam_Ratio'] = df['Work_Exp_Fam_Ratio'] * (1 - df['Work_Experience_Unknown']) * (1 - df['Family_Size_Unknown'])
  df['Age_Exp'] = df['Age_Exp'] * (1 - df['Work_Experience_Unknown'])
  df['Graduated'] = df['Graduated'] * (1 - df['Graduated_Unknown'])
  return df

data_path = "./data/train.parquet"
data = pd.read_parquet(data_path)
X_train = data.drop(columns="Segmentation")
#data = transform_data_rf(data)
colstest = list(X_train.columns)
kmeans = joblib.load('kmeans_model.pkl')
labels = kmeans.predict(X_train)

data['Cluster_Labels'] = labels

data = pd.get_dummies(data, columns=['Cluster_Labels'], prefix='Cluster')

cols =["Segmentation",
    "Segmentation_Encoded"]

label_encoder = LabelEncoder()
data["Segmentation_Encoded"] = label_encoder.fit_transform(data["Segmentation"])


rf_param_grid = {
    'n_estimators': [25,50,100, 200],       
    'max_depth': [2,4,8,10,16,20, None],       
    'min_samples_split': [2,3,5,7,13],          
    'bootstrap': [True, False]               
}

cols_cats=['Cat_1', 'Cat_2', 'Cat_3', 'Cat_4', 'Cat_5', 'Cat_6', 'Cat_7','Age_Unknown', 'Artist','Doctor', 'Engineer', 'Entertainment', 
  'Ever_Married', 'Ever_Married_Unknown', 'Executive', 'Gender', 'Graduated', 'Graduated_Unknown', 'Healthcare', 'Homemaker', 'Lawyer','Marketing',
  'Profession_Unknown','Var_1_Unknown','Cluster_0', 'Cluster_1','Cluster_2', 'Cluster_3']

cols_sel=['Age_Unknown', 'Artist', 'Cat_6', 'Doctor', 'Engineer', 'Entertainment', 
  'Ever_Married', 'Ever_Married_Unknown', 'Executive', 'Family_Size', 'Family_Size_Unknown',
  'Gender', 'Graduated', 'Graduated_Unknown', 'Healthcare', 'Homemaker', 'Lawyer', 'Log_Age', 'Log_Spending_Score_Age',
  'Marketing','Profession_Unknown', 'Spending_Score', 'Spending_Score_Age', 'Work_Experience_Unknown',
  'Cluster_0', 'Cluster_1','Cluster_2', 'Cluster_3',"Age_bin","W_X_bin"]

age_bins = [18, 30, 45, 60, float('inf')]
age_labels = [1,2,3,4]
data['Age_bin'] = pd.cut(data['Age'], bins=age_bins, labels=age_labels, include_lowest=True)
data=data.drop(columns=["Age"])

bins = [0, 3, 7, 10, float('inf')]
we_labels=[0,1,2,3]
data['W_X_bin'] = pd.cut(data['Work_Experience'], bins=bins,labels=we_labels, include_lowest=True)
data=data.drop(columns=["Work_Experience"])

data["W_X_bin"]=data["W_X_bin"].astype(float)
data['Age_bin']=data['Age_bin'].astype(float)


Y = data["Segmentation_Encoded"]
print('length of cols sel',len(cols_sel))
X = data[cols_sel]
dups = X.duplicated()
print("sum of dups ",sum(dups))
ndups = ~dups
X=X[ndups]
Y=Y[ndups]
print(len(Y))
custom_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

random_forest = RandomForestClassifier(random_state=0)

grid_search = GridSearchCV(random_forest, rf_param_grid, cv=custom_cv, scoring='accuracy',n_jobs=8)
grid_search.fit(X, Y)

print("best parameters :", grid_search.best_params_)
print("accuracy:", grid_search.best_score_)

best_rf = grid_search.best_estimator_

# feature_importances = best_rf.feature_importances_
predictions = cross_val_predict(best_rf, X, Y, cv=custom_cv)

accuracy = accuracy_score(Y, predictions)
conf_matrix = confusion_matrix(Y, predictions)
print("Confusion Matrix:\n", conf_matrix)
print(f"\nAccuracy: {accuracy:.4f}")


test_data_path = "./data/test.parquet"
test_data = pd.read_parquet(test_data_path)
X_test = test_data.drop(columns=["Segmentation"])
X_test=X_test[colstest]
test_cluster_labels = kmeans.predict(X_test)
test_data['Cluster_Labels'] = test_cluster_labels
test_data = pd.get_dummies(test_data, columns=['Cluster_Labels'], prefix='Cluster')
test_data["Segmentation_Encoded"] = label_encoder.transform(test_data["Segmentation"])
print('-'*40)
print(test_data["Segmentation"].value_counts())
print("-"*40)
test_data['Age_bin'] = pd.cut(test_data['Age'], bins=age_bins, labels=age_labels, include_lowest=True).astype(float)
test_data = test_data.drop(columns=["Age"])
test_data['W_X_bin'] = pd.cut(test_data['Work_Experience'], bins=bins, labels=we_labels, include_lowest=True).astype(float)
test_data = test_data.drop(columns=["Work_Experience"])

X_test=test_data[cols_sel]

test_dups = test_data.duplicated()
X_test = X_test[~test_dups]
Y_test = test_data["Segmentation_Encoded"]
Y_test=Y_test[~test_dups]
print("test shape", X_test.shape)
test_predictions = best_rf.predict(X_test)
test_conf_matrix = confusion_matrix(Y_test, test_predictions)
test_accuracy = accuracy_score(Y_test, test_predictions)
print("-"*40)
print("Test Confusion Matrix:\n", test_conf_matrix)
print(f"Test Accuracy: {test_accuracy:.4f}")