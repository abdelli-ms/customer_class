## General
dataset link : https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation/
## Steps:
* order of execution of scripts is important. (kmeans)  
* * run cleaning.py to get a clean dataset and cut missing data out (details in the file)
* * run features.py to infer new data and get insight into next steps for a better performance.
* * run viz_kmeans.py to get a better insight of the information in the dataset, also important for generation of kmeans model. 
* * run cv_rf.py to get the final trained algorithm.
## Data files: 
  * /data directory contains:
  * *  original datasets (or_*.csv)
  * *  first parquet file generated in first-cleaning (test_data.parquet, train_data.parquet)
  * *  second parquet file generated from the exploration-features (test.parquet, trian.parquet)
  * main directory:
  * * kmeans_model.pkl contains the pre classification model prior to execution of cv_rf.py
  * * rf.pkl contains the final random forest model.
## Remarks:  
- The code order and separation is not given in final product format, it is written to give a better insight of the analysis done through out the project.
- csv files containing information on the dataset are given for easy access.

