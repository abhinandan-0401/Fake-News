The input data for training and testing has to be downloaded from the following url:
https://www.kaggle.com/c/fake-news/data

Once input datasets are downloaded under this directory run the "docker_build_run_preprocess.sh" script. This script would preprocess the input training and testing datasets and store them under this directory with names "processed_train.csv" and "processed_test.csv".

We will train the sequence classification model on the "processed_train.csv" and final predictions would be made on the "processed_test.csv" file. 

The final predictions would be available in the "submit.csv" file.