import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("random_forest_bayes_diff.csv")
    nunique = df.nunique(axis=1)

    print(df)

    msk = np.random.rand(len(df)) <= 0.50
    rf = df.drop(["Class_bayes"], axis=1)
    rf = rf[msk]
    #rf = rf.drop(["Class_bayes"], axis=1)
    print(rf)

    b = df.drop(["Class"], axis=1)
    b = b[~msk]
    #b = b.drop(["Class"], axis=1)
    print(b)
    b.columns = ["ID", "Class"]
    ret = pd.concat([rf, b], axis=0)

    #df = df.mode(axis=1)

    ret.to_csv("bayes_rf_kaggle_2.csv", index=False)
    exit()
    mask = nunique == 1
    mask.columns = ["truth"]
    print(mask)
    print(mask.mean())
    valid = pd.concat([df, mask], axis=1, ignore_index=True)
    #valid = valid.drop([1, 2, 3], axis=1)
    #valid.columns = ["Class", "truth"]
    print(valid)
    #id = pd.DataFrame({"ID":range(750, 1500)})
    #valid = pd.concat([valid, id], axis=1)

    valid = valid.set_index(2)
    v = pd.DataFrame(valid.loc[True])
    msk = np.random.rand(len(v)) <= 0.50
    #v.reset_index()
    #v.set_index("ID")
    #print(v)
    #v.drop(["truth"])
    #valid = df[mask]
    v.to_csv("brf.csv", index = False)
    # x = pd.read_csv("brf.csv")
    # #x.drop(["truth"])
    # cols = list(x.columns.values) #Make a list of all of the columns in the df
    # cols.pop(cols.index('Class')) #Remove b from list
    # #cols.pop(cols.index('x')) #Remove x from list
    # x = x[cols+['Class']]
    #
    # x.to_csv("brf.csv")


main()
