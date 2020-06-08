
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



def job(i):

    results = pd.DataFrame()
    prediction_voting = pd.DataFrame()
    df_train = pd.read_csv("learning_set_full0.9.csv")
    #df_train =df_train.drop(["ID"], axis=1)
    y = df_train["Class"]
    X = df_train.drop(['Class'], axis=1)
    df_test = pd.read_csv("data/amazon_test.csv")
    df_test = df_test.drop(["ID"], axis=1)
    X_p = df_test

    alphas = [1]

    for alpha in alphas:
        result_row = {}

        result_row["fold"] = i
        result_row["alpha"] = alpha

        rf = MultinomialNB(alpha=i)
        rf.fit(X, y)
        predicted = rf.predict(X_p)
        predicted_df = pd.DataFrame(predicted)
        predicted_df.columns = ["Class_"+str(alpha)]
        prediction_voting = pd.concat([prediction_voting, predicted_df], axis=1)
        predicted_df.to_csv("predicted_amazon_bayes_09.csv", sep=",", index=False)
        #result_row["score"] = round(rf.score(X_p,df_test["Class"]), 4)
        #confusion = confusion_matrix(df_test["Class"], predicted)
        #conf = pd.DataFrame(confusion)
        #conf.to_csv(target_path+"confusion_"+str(i)+"_"+str(alpha)+".csv", index=False)
        results = results.append(result_row, ignore_index = True)
    return prediction_voting

def main():

    results = pd.DataFrame()

    num_cores = multiprocessing.cpu_count()

    arr_results = Parallel(n_jobs=num_cores)(delayed(job)(i)  for i in [0.161])
    for a in arr_results:
        results = pd.concat([results, a], axis=1)

    results.to_csv("prediction_voting_bayes.csv")
    exit()
    results = results.astype({"fold": int})
    print(results)
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

    results_fold = results.set_index(["alpha"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])


    value_mapping = {}
    value_mapping["alpha"] =  [0.17]

    means = []
    maxs = []
    for j in value_mapping["alpha"]:
        fold = results_fold.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)
    print(maxs)


    print(results_fold.index.unique().tolist())
    print(results_fold["score"].mean(axis=0))
    plt.scatter(results_fold.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("mean score for smoothing parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/alpha_mean.png')
    plt.close()

    plt.scatter(results_fold.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("max score for smoothing parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xscale("log")
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/alpha_max.png')
    plt.close()

    print(results)
    results.to_csv(target_path+"resuts.csv", index=True)
    print("done")

    return True


main()
