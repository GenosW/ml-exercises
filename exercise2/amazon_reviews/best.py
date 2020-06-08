import pandas as pd
import numpy as np


def main():

    # df_rt = pd.read_csv("vote_all.csv")
    # nunique = df_rt.nunique(axis=1)
    #
    # print(df_rt)
    #
    # df_rt = df_rt.mode(axis=1)
    # df_rt.to_csv("vote_all_mode.csv", index = False)

    df_rt = pd.read_csv("prediction_voting.csv")
    nunique = df_rt.nunique(axis=1)

    print(df_rt)

    df_rt = df_rt.mode(axis=1)
    df_rt.to_csv("vote_rt_mode.csv", index = False)

    df_by = pd.read_csv("prediction_voting_bayes.csv")
    nunique = df_by.nunique(axis=1)

    print(df_by)

    df_by = df_by.mode(axis=1)
    df_by.to_csv("vote_bayes_mode.csv", index = False)

    df_rt = pd.read_csv("prediction_voting.csv")
    df_by = pd.read_csv("prediction_voting_bayes.csv")

    df = pd.concat([df_by, df_rt], axis=1)
    df = df.mode(axis=1)
    df.to_csv("vote_all.csv", index = False)


    df = pd.read_csv("votes.csv")
    nunique = df.nunique(axis=1)

    print(df)

    df = df.mode(axis=1)

    df.to_csv("votes_kaggle.csv")
    exit()
    mask = nunique == 2
    mask.columns = ["truth"]
    print(mask)
    print(mask.mean())
    valid = pd.concat([df, mask], axis=1)
    valid = valid.drop([1, 2, 3], axis=1)



    valid.columns = ["Class", "truth"]
    print(valid)
    #id = pd.DataFrame({"ID":range(750, 1500)})
    #valid = pd.concat([valid, id], axis=1)
    test = pd.read_csv("data/amazon_test.csv")
    valid = pd.concat([valid, test], axis=1)

    valid = valid.set_index(["truth"])
    v = pd.DataFrame(valid.loc[True])
    #v.reset_index()
    #v.set_index("ID")
    #print(v)
    #v.drop(["truth"])
    #valid = df[mask]
    v.to_csv("valid.csv", index = False)
    x = pd.read_csv("valid.csv")
    #x.drop(["truth"])
    cols = list(x.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('Class')) #Remove b from list
    #cols.pop(cols.index('x')) #Remove x from list
    x = x[cols+['Class']]

    x.to_csv("valid.csv")


main()
