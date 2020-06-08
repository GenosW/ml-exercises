import pandas as pd
import sys
import numpy as np


np.random.seed(1645)
df = pd.read_csv('preprocessed.csv', dtype="str")

msk = np.random.rand(len(df)) <= 0.50
# Split the data into train and test

train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_1.csv' ,   index=False)
test.to_csv('preprocessed_test_1.csv',   index=False)

np.random.seed(1234)
msk = np.random.rand(len(df)) <= 0.5
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_2.csv' ,   index=False)
test.to_csv('preprocessed_test_2.csv',   index=False)

np.random.seed(145)
msk = np.random.rand(len(df)) <= 0.5
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_3.csv' ,   index=False)
test.to_csv('preprocessed_test_3.csv',   index=False)
np.random.seed(45)
msk = np.random.rand(len(df)) <= 0.5
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_4.csv' ,   index=False)
test.to_csv('preprocessed_test_4.csv',   index=False)
np.random.seed(468)
msk = np.random.rand(len(df)) <= 0.50
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_5.csv' ,   index=False)
test.to_csv('preprocessed_test_5.csv',   index=False)
np.random.seed(497)
msk = np.random.rand(len(df)) <= 0.50
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_6.csv' ,   index=False)
test.to_csv('preprocessed_test_6.csv',   index=False)
np.random.seed(9506)
msk = np.random.rand(len(df)) <= 0.50
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_7.csv' ,   index=False)
test.to_csv('preprocessed_test_7.csv',   index=False)
np.random.seed(4785)
msk = np.random.rand(len(df)) <= 0.50
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_8.csv' ,   index=False)
test.to_csv('preprocessed_test_8.csv',   index=False)
np.random.seed(7315)
msk = np.random.rand(len(df)) <= 0.50
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_9.csv' ,   index=False)
test.to_csv('preprocessed_test_9.csv',   index=False)
np.random.seed(36874)
msk = np.random.rand(len(df)) <= 0.50
train = df[msk]
test = df[~msk]
train.to_csv('preprocessed_training_10.csv' ,   index=False)
test.to_csv('preprocessed_test_10.csv',   index=False)




sys.exit()
