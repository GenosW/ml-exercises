
import argparse
import pandas as pd
import numpy as np
import sys
import os
import sklearn as sk
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


target_path = "target/adaboost/"

n_estimators = [ 4001]
learning_rate = [ 0.01]

def job(i):
    results = pd.DataFrame()
    df_train = pd.read_csv("preprocessed.csv")
    df_train = df_train.drop(["ID"], axis=1)
    y = df_train["Class"]
    X = df_train.drop(['Class'], axis=1)
    df_test = pd.read_csv("data/amazon_test.csv")
    df_test = df_test.drop(["ID"], axis=1)
    X_p = df_test#df_test.drop(["Class"], axis=1)

    global n_estimators
    #n_estimators = [1, 8]
    global learning_rate
    #learning_rate = [1, 2]
    random_state = 1428
    min_samples_split = [2]
    min_samples_leaf = [1]
    for n in n_estimators:
        for d in learning_rate:
            for s in min_samples_split:
                for l in min_samples_leaf:
                    result_row = {}
                    result_row["n_estimators"] = n
                    result_row["fold"] = i
                    result_row["learning_rate"] = d
                    result_row["min_samples_split"] = s
                    result_row["min_samples_leaf"] = l
                    print(result_row)

                    rf = AdaBoostClassifier(base_estimator=None, n_estimators=n, learning_rate=d, algorithm='SAMME.R', random_state=None)
                    rf.fit(X, y)
                    predicted = rf.predict(X_p)
                    predicted_df = pd.DataFrame(predicted)
                    predicted_df.to_csv("predicted_amazon_adaboost_"+str(d)+".csv", sep=",", index=False)
                    #result_row["score"] = round(rf.score(X_p,df_test["Class"]), 4)
                    #confusion = confusion_matrix(df_test["Class"], predicted)
                    #conf = pd.DataFrame(confusion)
                    #conf.to_csv(target_path+"confusion_"+str(i)+"_"+str(n)+"_"+str(d)+"_"+str(s)+"_"+str(l)+"_"+".csv", index=False)
                    results = results.append(result_row, ignore_index = True)
    return results

def main():

    results = pd.DataFrame()

    num_cores = multiprocessing.cpu_count()

    arr_results = Parallel(n_jobs=num_cores)(delayed(job)(i)  for i in range (1, 2))
    for a in arr_results:
        results = results.append(a)

    results.to_csv(target_path+"results.csv", index=True)
    results = results.astype({"fold": int})
    results_fold = results.set_index(["fold"])
    #results = results.set_index(["fold", "n_estimators", "learning_rate", "min_samples_split", "min_samples_leaf"])

    value_mapping = {}
    value_mapping["fold"] = ["", "0.95", "0.85", "0.75", "0.65", "0.55", "0.45", "0.35", "0.25", "0.15", "0.05"]

    means = []
    maxs = []
    for j in range(1,11):
        fold = results_fold.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)


    print(results_fold.index.unique().tolist())
    print(results_fold["score"].mean(axis=0))
    plt.scatter(results_fold.index.unique().tolist(), means)
    plt.xticks(range(len(value_mapping["fold"])))
    plt.title("mean score for training ratio")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    plt.axes().set_xticklabels(value_mapping["fold"])
    plt.savefig(target_path+'plots/k_fold.png')
    plt.close()

    plt.scatter(results_fold.index.unique().tolist(), maxs)
    plt.xticks(range(len(value_mapping["fold"])))
    plt.title("max score for training ratio")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    plt.axes().set_xticklabels(value_mapping["fold"])
    plt.savefig(target_path+'plots/k_fold_max.png')
    plt.close()


    results_n_estimators = results.set_index(["n_estimators"])
    #results = results.set_index(["fold", "n_estimators", "learning_rate", "min_samples_split", "min_samples_leaf"])
    global n_estimators
    value_mapping = {}
    value_mapping["n_estimators"] = n_estimators

    means = []
    maxs = []
    for j in value_mapping["n_estimators"]:
        fold = results_n_estimators.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)


    print(results_n_estimators.index.unique().tolist())
    print(results_n_estimators["score"].mean(axis=0))
    plt.scatter(results_n_estimators.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["n_estimators"])))
    plt.title("mean score for n estimators")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["n_estimators"])
    plt.savefig(target_path+'plots/n_estimators.png')
    plt.close()

    plt.scatter(results_n_estimators.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["n_estimators"])))
    plt.title("max score for n estimators")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["n_estimators"])
    plt.savefig(target_path+'plots/n_estimators_max.png')
    plt.close()


    results_n_estimators = results.set_index(["learning_rate"])
    #results = results.set_index(["fold", "n_estimators", "learning_rate", "min_samples_split", "min_samples_leaf"])

    value_mapping = {}
    global learning_rate
    value_mapping["learning_rate"] = learning_rate

    means = []
    maxs = []
    for j in value_mapping["learning_rate"]:
        fold = results_n_estimators.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)


    print(results_n_estimators.index.unique().tolist())
    print(results_n_estimators["score"].mean(axis=0))
    plt.scatter(results_n_estimators.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["learning_rate"])))
    plt.title("mean score for max depth")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["learning_rate"])
    plt.savefig(target_path+'plots/learning_rate.png')
    plt.close()

    plt.scatter(results_n_estimators.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["learning_rate"])))
    plt.title("max score for max depth")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["learning_rate"])
    plt.savefig(target_path+'plots/learning_rate_max_score.png')
    plt.close()

    print(results)

    print("done")

    return True


main()
