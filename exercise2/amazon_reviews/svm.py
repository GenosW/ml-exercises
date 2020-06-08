
import argparse
import pandas as pd
import numpy as np
import sys
import os
import sklearn as sk
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
import multiprocessing


target_path = "target/svm/"



def job(i):

    results = pd.DataFrame()
    df_train = pd.read_csv("preprocessed_training_"+str(i)+".csv")
    df_train = df_train.drop(["ID"], axis=1)
    y = df_train["Class"]
    X = df_train.drop(['Class'], axis=1)
    df_test = pd.read_csv("preprocessed_test_"+str(i)+".csv")
    df_test = df_test.drop(["ID"], axis=1)
    X_p = df_test.drop(["Class"], axis=1)

    C = [ 0.5, 1, 2, 3, 4, 10 ]
    kernel = ["linear", "poly", "rbf", "sigmoid"]
    degree = [ 2, 3, 4 ]

    for c in C:
        for k in kernel:
            if(k == "poly"):

                for d in degree:

                    print("fold"+str(i)+"_"+str(c)+"_"+str(k)+"_"+str(d))

                    result_row = {}

                    result_row["fold"] = i
                    result_row["C"] = c
                    result_row["kernel"] = k
                    result_row["degree"] = d

                    rf = SVC(C=c, kernel = k, degree = d)
                    rf.fit(X, y)
                    predicted = rf.predict(X_p)
                    result_row["score"] = round(rf.score(X_p,df_test["Class"]), 4)
                    confusion = confusion_matrix(df_test["Class"], predicted)
                    conf = pd.DataFrame(confusion)
                    conf.to_csv(target_path+"confusion_"+str(i)+"_"+str(c)+"_"+str(k)+"_"+str(d)+".csv", index=False)
                    results = results.append(result_row, ignore_index = True)
            else:
                print("fold"+str(i)+"_"+str(c)+"_"+str(k))

                result_row = {}

                result_row["fold"] = i
                result_row["C"] = c
                result_row["kernel"] = k
                result_row["degree"] = 0

                rf = SVC(C=c, kernel = k)
                rf.fit(X, y)
                predicted = rf.predict(X_p)
                result_row["score"] = round(rf.score(X_p,df_test["Class"]), 4)
                confusion = confusion_matrix(df_test["Class"], predicted)
                conf = pd.DataFrame(confusion)
                conf.to_csv(target_path+"confusion_"+str(i)+"_"+str(c)+"_"+str(k)+"_"+str(0)+".csv", index=False)
                results = results.append(result_row, ignore_index = True)
    return results

def main():

    results = pd.DataFrame()

    num_cores = multiprocessing.cpu_count()

    arr_results = Parallel(n_jobs=num_cores)(delayed(job)(i)  for i in range (1, 11))
    for a in arr_results:
        results = results.append(a)

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


    results_fold = results.set_index(["C"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])


    value_mapping = {}
    value_mapping["C"] = [ 0.5, 1, 2, 3, 4, 10 ]
    means = []
    maxs = []
    for j in value_mapping["C"]:
        fold = results_fold.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)
    print(maxs)


    print(results_fold.index.unique().tolist())
    print(results_fold["score"].mean(axis=0))
    plt.scatter(results_fold.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("mean score for C parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/C_mean.png')
    plt.close()

    plt.scatter(results_fold.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("max score for C parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xscale("log")
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/C_max.png')
    plt.close()


    results_fold = results.set_index(["kernel"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])
    value_mapping = {}
    value_mapping["kernel"] = ["linear", "poly", "rbf", "sigmoid"]
    means = []
    maxs = []
    for j in value_mapping["kernel"]:
        fold = results_fold.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)
    print(maxs)


    print(results_fold.index.unique().tolist())
    print(results_fold["score"].mean(axis=0))
    plt.scatter(results_fold.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("mean score for kernel parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/kernel_mean.png')
    plt.close()

    plt.scatter(results_fold.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("max score for kernel parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xscale("log")
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/kernel_max.png')
    plt.close()


    results_fold = results.set_index(["degree"])
    #results = results.set_index(["fold", "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"])
    value_mapping = {}
    value_mapping["degree"] =  [0, 2, 3, 4]
    means = []
    maxs = []
    for j in value_mapping["degree"]:
        fold = results_fold.loc[j]
        means.append(fold.mean()["score"])
        maxs.append(fold.max()["score"])
    print(means)
    print(maxs)


    print(results_fold.index.unique().tolist())
    print(results_fold["score"].mean(axis=0))
    plt.scatter(results_fold.index.unique().tolist(), means)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("mean score for degree parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/degree_mean.png')
    plt.close()

    plt.scatter(results_fold.index.unique().tolist(), maxs)
    #plt.xticks(range(len(value_mapping["alpha"])))
    plt.title("max score for degree parameter")
    plt.ylabel("score")
    plt.align='center'

    ax = plt.axes()
    #plt.axes().set_xscale("log")
    #plt.axes().set_xticklabels(value_mapping["alpha"])
    plt.savefig(target_path+'plots/degree_max.png')
    plt.close()

    print(results)
    results.to_csv(target_path+"resuts.csv", index=True)
    print("done")

    return True


main()
