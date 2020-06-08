
import argparse
import pandas as pd
import numpy as np
import sys
import os
import sklearn as sk
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


target_path = "target/bayes/"

alphas = [ .1615,.1605 ]

def job(i):

    results = pd.DataFrame()


    df_train = pd.read_csv("learning_set_full"+str(i)+".csv")
    #df_train["Class"] = df_train["V10001"]
    #df_train = df_train.drop(["V10001"], axis=1)
    #df_train = df_train.drop(["ID"], axis=1)
    y = df_train["Class"]
    print(y)
    X = df_train.drop(['Class'], axis=1)
    X_1 = X
    y1 = list(y)
    df_test = pd.read_csv("data_test_verification.csv")
    #X_1["Class"] = pd.DataFrame(y1)
    X_p = df_test.drop(["Class"], axis=1)

    global alphas
    last_score = 0
    for e in range(1,2):
        for index, row in X.iterrows():

            X_1_old = X_1
            y1_old = y1
            X_1 = X_1.append(row)
            y1.append(y.iloc[index])
            print(last_score)

            rf = MultinomialNB(alpha=i)
            rf.fit(X_1, y1)
            predicted = rf.predict(X_p)
            #print(predicted)
            new_score = round(rf.score(X_p,df_test["Class"]), 12)
            if last_score>new_score:
                X_1 = X_1_old
                y1.pop()
            else:
                last_score=new_score
    X_1["Class"] = pd.DataFrame(y1)
    X_1.to_csv("learning_set_full"+str(i)+".csv", index=False)
    print(i)
    print(last_score)
    return last_score

def main():

    results = pd.DataFrame()

    num_cores = multiprocessing.cpu_count()

    arr_results = Parallel(n_jobs=7)(delayed(job)(i)  for i in [0.1, 0.161, 0.40, 0.9, 1, 2])
    print(arr_results)


main()
