
import argparse
import pandas as pd
import numpy as np
import sys
import os
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


target_path = "target/random_forest/"

n_estimators = [  16005, 20001]
max_depth = [ 17, 37]

def job(i):
    results = pd.DataFrame()
    df_train = pd.read_csv("data/amazon_full_2.csv")
    df_train["Class"] = df_train["V10001"]
    df_train = df_train.drop(["V10001"], axis=1)
    #df_train = df_train.drop(["ID"], axis=1)
    df_train = df_train.drop(["ID"], axis=1)
    y = df_train["Class"]
    X = df_train.drop(['Class'], axis=1)
    df_test = pd.read_csv("data_test_verification.csv")
    #df_test = df_test.drop(["ID"], axis=1)
    X_p = df_test.drop(["Class"], axis=1)

    n_estimators = [2]
    #n_estimators = [1, 8]
    global max_depth
    #max_depth = [1, 2]
    random_state = 1428
    min_samples_split = [2]
    min_samples_leaf = [1]
    for n in n_estimators:
        for d in max_depth:
            for s in min_samples_split:
                for l in min_samples_leaf:
                    result_row = {}
                    result_row["n_estimators"] = i
                    result_row["fold"] = i
                    result_row["max_depth"] = d
                    result_row["min_samples_split"] = s
                    result_row["min_samples_leaf"] = l

                    rf = RandomForestClassifier(n_estimators=i, max_depth=d, random_state=random_state, criterion="gini", min_samples_split=s, min_samples_leaf=l)
                    rf.fit(X, y)
                    predicted = rf.predict(X_p)
                    result_row["score"] = round(rf.score(X_p,df_test["Class"]), 4)
                    confusion = confusion_matrix(df_test["Class"], predicted)
                    conf = pd.DataFrame(confusion)
                    conf.to_csv(target_path+"confusion_"+str(i)+"_"+str(n)+"_"+str(d)+"_"+str(s)+"_"+str(l)+"_"+".csv", index=False)
                    results = results.append(result_row, ignore_index = True)
    return results

def main():
    global n_estimators

    results = pd.DataFrame()

    num_cores = multiprocessing.cpu_count()

    arr_results = Parallel(n_jobs=num_cores)(delayed(job)(i)  for i in n_estimators)
    for a in arr_results:
        results = results.append(a)

    results.to_csv(target_path+"results.csv", index=True)
    results = results.astype({"fold": int})
    results_fold = results.set_index(["fold"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])

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
    plt.savefig('./target/random_forest/plots/k_fold.png')
    plt.close()

    plt.scatter(results_fold.index.unique().tolist(), maxs)
    plt.xticks(range(len(value_mapping["fold"])))
    plt.title("max score for training ratio")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    plt.axes().set_xticklabels(value_mapping["fold"])
    plt.savefig('./target/random_forest/plots/k_fold_max.png')
    plt.close()


    results_n_estimators = results.set_index(["n_estimators"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])
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
    plt.savefig('./target/random_forest/plots/n_estimators.png')
    plt.close()

    plt.scatter(results_n_estimators.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["n_estimators"])))
    plt.title("max score for n estimators")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["n_estimators"])
    plt.savefig('./target/random_forest/plots/n_estimators_max.png')
    plt.close()


    results_n_estimators = results.set_index(["max_depth"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])

    value_mapping = {}
    global max_depth
    value_mapping["max_depth"] = max_depth

    means = []
    maxs = []
    for j in value_mapping["max_depth"]:
        fold = results_n_estimators.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)


    print(results_n_estimators.index.unique().tolist())
    print(results_n_estimators["score"].mean(axis=0))
    plt.scatter(results_n_estimators.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["max_depth"])))
    plt.title("mean score for max depth")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["max_depth"])
    plt.savefig('./target/random_forest/plots/max_depth.png')
    plt.close()

    plt.scatter(results_n_estimators.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["max_depth"])))
    plt.title("max score for max depth")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["max_depth"])
    plt.savefig('./target/random_forest/plots/max_depth_max_score.png')
    plt.close()


    results_n_estimators = results.set_index(["min_samples_split"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])

    value_mapping = {}
    value_mapping["min_samples_split"] = [2]

    means = []
    for j in value_mapping["min_samples_split"]:
        fold = results_n_estimators.loc[j]
        means.append(fold.mean()["score"])
    print(means)


    print(results_n_estimators.index.unique().tolist())
    print(results_n_estimators["score"].mean(axis=0))
    plt.scatter(results_n_estimators.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["min_samples_split"])))
    plt.title("mean score for min_samples_split")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["min_samples_split"])
    plt.savefig('./target/random_forest/plots/min_samples_split.png')
    plt.close()

    results_n_estimators = results.set_index(["min_samples_leaf"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])

    value_mapping = {}
    value_mapping["min_samples_leaf"] = [1]

    means = []
    for j in value_mapping["min_samples_leaf"]:
        fold = results_n_estimators.loc[j]
        means.append(fold.mean()["score"])
    print(means)


    print(results_n_estimators.index.unique().tolist())
    print(results_n_estimators["score"].mean(axis=0))
    plt.scatter(results_n_estimators.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["min_samples_leaf"])))
    plt.title("mean score for min_samples_leaf")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["min_samples_leaf"])
    plt.savefig('./target/random_forest/plots/min_samples_leaf.png')
    plt.close()

    print(results)

    print("done")

    return True


main()
