import pandas as pd
import numpy as np


def main():
    df = pd.read_csv("predicted_amazon_bayes_09.csv")
    nunique = df.nunique(axis=1)

    print(df)

    df = df.mode(axis=1)

    df.to_csv("bayes_kaggle_2.csv")
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
