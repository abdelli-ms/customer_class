import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

def plot_column_distributions(df, columns, hue=None,cols_per_row=3):
  rows = math.ceil(len(columns) / cols_per_row)
  fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 6, rows * 4))
  axes = axes.flatten()
  
  for i, column in enumerate(columns):
    sns.histplot(data=df, x=column, hue=hue,kde=True, ax=axes[i])
    axes[i].set_title(f'Distribution of {column}')
  
  for j in range(len(columns), len(axes)):
    axes[j].axis('off')
  
  plt.tight_layout()
  plt.show()


def analyze_cats(df, binary_cols, target_col):
  for col in binary_cols:
    print(f"Analysis for binary column: {col}")
    
    counts = df.groupby([col, target_col],observed=False).size().unstack(fill_value=0)
    print(counts)
    counts.plot(kind='bar', stacked=True, figsize=(8, 5), colormap='viridis')
    plt.title(f"Target distribution by binary column {col}")
    plt.xlabel(f"{col} (0 or 1)")
    plt.ylabel("Count")
    plt.legend(title=target_col)
    plt.tight_layout()
    plt.show()
    import pandas as pd


def print_value_counts(df, columns):
  for col in columns:
    print(f"Value counts for column: {col}")
    print(df[col].value_counts())
    print("-" * 40) 
def print_value_counts_by_target(df, columns, target_col):
  for col in columns:
    print(f"Value counts for column: {col} grouped by {target_col}")
    counts = df.groupby(target_col)[col].value_counts().unstack(fill_value=0)
    print(counts)
    print("-" * 40)
def correlation(df, output_csv_path):
    correlation_matrix = df.corr()
    correlation_matrix.to_csv(output_csv_path, index=True, header=True)
import seaborn as sns
import matplotlib.pyplot as plt

def plot_correlation_heatmap(correlation_matrix):
  n = len(correlation_matrix.columns)
  
  first_half = correlation_matrix.iloc[:, :n//2]
  second_half = correlation_matrix.iloc[:, n//2:]

  plt.figure(figsize=(10, 8))
  sns.heatmap(first_half, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
  plt.title("Correlation Heatmap - First Half")
  plt.show()

  plt.figure(figsize=(10, 8))
  sns.heatmap(second_half, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
  plt.title("Correlation Heatmap - Second Half")
  plt.show()


data_path="./data/train.parquet"
data = pd.read_parquet(data_path)
cols_age=["Age","Log_Age","Age_2","Age_Normalized","Log_Age_Normalized"]
cols_work=["Work_Experience","Work_Experience_2","Log_Work_Experience"]
cat_s=["Cat_1","Cat_2","Cat_3","Cat_4","Cat_5","Cat_6","Ever_Married","Ever_Married_Unknown",
"Artist","Doctor",'Engineer', 'Entertainment', 'Executive','Gender', 'Graduated','Graduated_Unknown',
'Healthcare', 'Homemaker', 'Lawyer','Marketing', 'Profession_Unknown']
cols_fam=["Family_Size","Family_Size_2"]
cols_WA=["Log_Age_Exp","Age_Exp","Age_Exp_2"]
cols_spending=["Spending_Score_Age", "Spending_Score_Age_2", "Log_Spending_Score_Age"]

kmeans = KMeans(n_clusters=4, random_state=0)
X=data.drop(columns="Segmentation")
cluster_labels = kmeans.fit_predict(X)
X['Cluster_Labels'] = cluster_labels
joblib.dump(kmeans, 'kmeans_model.pkl')

#decomment for figures
plot_column_distributions(data, columns=cols_age)
# "Normalized Age"

# work experience
plot_column_distributions(data,columns=cols_work)

#fam size
# plot_column_distributions(data,columns=cols_fam)


#work age
# plot_column_distributions(data,columns=cols_WA)

#spending
# plot_column_distributions(data,columns=cols_spending)

print_value_counts_by_target(data,columns=["Spending_Score"],target_col="Segmentation")
print("numuber of peple with 0 working experience:",(data['Work_Experience'] == 0).sum())
print("number of missing values for experience:", (data['Work_Experience_Unknown'] == 1.0).sum())
print("Number of missing values for Age:", (data['Age_Unknown'] == 1.0).sum())
print("Number of missing values for Ever Married:", (data['Ever_Married_Unknown'] == 1.0).sum())
print("Number of missing values for Family Size:", (data['Family_Size_Unknown'] == 1.0).sum())
print("Number of missing values for Graduated:", (data['Graduated_Unknown'] == 1.0).sum())
print("Number of missing values for Profession:", (data['Profession_Unknown'] == 1.0).sum())
print("Number of missing values for Var_1:", (data['Var_1_Unknown'] == 1.0).sum())
print("Number of missing values for Work Experience:", (data['Work_Experience_Unknown'] == 1.0).sum())


# 
encoder = LabelEncoder()
X["Cluster_Labels"] = encoder.fit_transform(X["Cluster_Labels"])

correlation_matrix = X.corr()
plot_correlation_heatmap(correlation_matrix)
print(X.columns)