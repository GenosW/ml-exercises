
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



def job(i):
    results = pd.DataFrame()
    prediction_voting = pd.DataFrame()
    df_train = pd.read_csv("data/amazon_full_2.csv")
    df_train["Class"] = df_train["V10001"]
    df_train = df_train.drop(["V10001"], axis=1)
    #df_train =df_train.drop(["ID"], axis=1)

    y = df_train["Class"]
    X = df_train.drop(['Class'], axis=1)
    #print(X)
    df_test = pd.read_csv("data/amazon_test.csv")
    df_test = df_test.drop(["ID"], axis=1)
    X_p = df_test

    n_estimators = [127, 1025, 16001]
    #n_estimators = [1, 8]
    max_depth = [2]
    #max_depth = [1, 2]
    random_states = [ 2511]
    min_samples_split = [2]
    min_samples_leaf = [1]
    for n in n_estimators:
        for d in max_depth:
            for s in min_samples_split:
                for l in min_samples_leaf:

                    result_row = {}
                    result_row["n_estimators"] = n
                    result_row["fold"] = i
                    result_row["max_depth"] = d
                    result_row["min_samples_split"] = s
                    result_row["min_samples_leaf"] = l
                    result_row["random_state"] = i

                    rf = RandomForestClassifier(n_estimators=n, max_depth=i, random_state=i, criterion="gini", min_samples_split=s, min_samples_leaf=l)
                    rf.fit(X, y)
                    predicted = rf.predict(X_p)
                    predicted_df = pd.DataFrame(predicted)
                    predicted_df.columns = ["Class_"+str(n)+"_"+str(i)+"_"+str(d)]
                    prediction_voting = pd.concat([prediction_voting, predicted_df], axis=1)
                    predicted_df.to_csv("predicted_amazon_forest_"+str(n)+"_"+str(d)+"_"+str(i)+".csv", sep=",", index=False)
                        #result_row["score"] = round(rf.score(X_p,df_test["Class"]), 4)
                        #confusion = confusion_matrix(df_test["Class"], predicted)
                        #conf = pd.DataFrame(confusion)
                        #conf.to_csv(target_path+"test_confusion_"+str(i)+"_"+str(n)+"_"+str(d)+"_"+str(s)+"_"+str(l)+"_"+".csv", index=False)
                    results = results.append(result_row, ignore_index = True)
    return prediction_voting

def main():

    results = pd.DataFrame()

    num_cores = multiprocessing.cpu_count()

    arr_results = Parallel(n_jobs=num_cores)(delayed(job)(i)  for i in [ 17, 37])
    for a in arr_results:
        results = pd.concat([results, a], axis=1)

    results.to_csv("prediction_voting.csv")
    exit()
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
    value_mapping["n_estimators"] = [101, 129, 171,  257, 1025, 2049]

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
    value_mapping["max_depth"] = [  32, 64, 128, 256]

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
    results.to_csv(target_path+"results.csv", index=True)
    print("done")

    return True


main()
