import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sl
from sklearn.preprocessing import OneHotEncoder

import arff


def count_param(row, param):
    print(row)
    return row.lower().count(param)

def label(row, param):
    return param[int(row)]

def is_class(row, param):
    return int(int(row) == param)

def is_equal_str(row, param):
    return int(str(row) == str(param))

def is_present(row):
    return int(int(row)>0)

def read_special_list(string):
    lst = string.split("||")
    lst2 = []
    for x in lst:
        x = x[x.rfind(":")+1:]
        lst2.append(x)
    return lst2

def get_data_pure():
    return pd.read_csv('data/flags.csv', dtype="str")

def get_test_data():
     #full_df = pd.DataFrame(arff.load(open('data/Amazon_initial_50_30_10000.arff', 'rb')))#
     full_df = pd.read_csv('data/amazon_full_2.csv')
     learn_df = pd.read_csv('data/amazon_learn.csv')
     test_df = pd.read_csv('data/amazon_test.csv')
     print(full_df.dtypes)
     #full_df["ID"] = range(0, 1500)
     full_df["Class"] = full_df["V10001"]
     full_df = full_df.drop(["V10001"], axis=1)
     learn_df = learn_df.drop(["ID"], axis=1)
     test_df = test_df.drop(["ID"], axis=1)
     print(full_df)
     #print(full_df["Class"].nunique())
     print(learn_df)


     test_answers = pd.concat([full_df, learn_df]).drop_duplicates(keep=False)
     test_df["Class"] = learn_df["Class"]
     test_data = pd.concat([full_df, test_df]).drop_duplicates(keep=False)
     test_data = test_data[test_data.duplicated(test_data.columns[:-1])]
     test_answers.to_csv("data_test_verification.csv", sep=',', index=False)
     test_data.to_csv("test_data.csv", sep=',', index=False)

def get_data_preprocessed():

    ds = get_data_pure()

    value_mapping = {}
    value_mapping["landmass"] = ["", "N. America",  "S. America", "Europe", "Africa", "Asia", "Oceania"]
    value_mapping["zone"] = ["",  "NE", "SE", "SW", "NW"]
    value_mapping["language"] = ["",  "English", "Spanish", "French", "German", "Slavic", "Other Indo-European", "Chinese", "Arabic", "Japanese/Turkish/Finnish/Magyar", "Others"]
    value_mapping["religion"] = ["Catholic",  "Other Christian", "Muslim", "Buddhist", "Hindu", "Ethnic", "Marxist", "Others"]
    value_mapping["red"] = ["absent",  "present"]
    value_mapping["green"] = ["absent",  "present"]
    value_mapping["blue"] = ["absent",  "present"]
    value_mapping["gold"] = ["absent",  "present"]
    value_mapping["white"] = ["absent",  "present"]
    value_mapping["black"] = ["absent",  "present"]
    value_mapping["orange"] = ["absent",  "present"]
    value_mapping["crescent"] = ["absent",  "present"]
    value_mapping["triangle"] = ["absent",  "present"]
    value_mapping["icon"] = ["absent",  "present"]
    value_mapping["animate"] = ["absent",  "present"]
    value_mapping["text"] = ["absent",  "present"]

    #one-hot encoding
    ds["N. America"] = ds.apply(lambda row: is_class(row["landmass"], 1), axis=1)
    ds["S. America"] = ds.apply(lambda row: is_class(row["landmass"], 2), axis=1)
    ds["Europe"] = ds.apply(lambda row: is_class(row["landmass"], 3), axis=1)
    ds["Africa"] = ds.apply(lambda row: is_class(row["landmass"], 4), axis=1)
    ds["Asia"] = ds.apply(lambda row: is_class(row["landmass"], 5), axis=1)
    ds["Oceania"] = ds.apply(lambda row: is_class(row["landmass"], 6), axis=1)

    ds["NE"] = ds.apply(lambda row: is_class(row["zone"], 1), axis=1)
    ds["SE"] = ds.apply(lambda row: is_class(row["zone"], 2), axis=1)
    ds["SW"] = ds.apply(lambda row: is_class(row["zone"], 3), axis=1)
    ds["NW"] = ds.apply(lambda row: is_class(row["zone"], 4), axis=1)

    ds["English"] = ds.apply(lambda row: is_class(row["language"], 1), axis=1)
    ds["Spanish"] = ds.apply(lambda row: is_class(row["language"], 2), axis=1)
    ds["French"] = ds.apply(lambda row: is_class(row["language"], 3), axis=1)
    ds["German"] = ds.apply(lambda row: is_class(row["language"], 4), axis=1)
    ds["Slavic"] = ds.apply(lambda row: is_class(row["language"], 5), axis=1)
    ds["Other Indo-European"] = ds.apply(lambda row: is_class(row["language"], 6), axis=1)
    ds["Chinese"] = ds.apply(lambda row: is_class(row["language"], 7), axis=1)
    ds["Arabic"] = ds.apply(lambda row: is_class(row["language"], 8), axis=1)
    ds["Japanese/Turkish/Finnish/Magyar"] = ds.apply(lambda row: is_class(row["language"], 9), axis=1)
    ds["Others"] = ds.apply(lambda row: is_class(row["language"], 10), axis=1)

    ds["Catholic"] = ds.apply(lambda row: is_class(row["religion"], 0), axis=1)
    ds["Other Christian"] = ds.apply(lambda row: is_class(row["religion"], 1), axis=1)
    ds["Muslim"] = ds.apply(lambda row: is_class(row["religion"], 2), axis=1)
    ds["Buddhist"] = ds.apply(lambda row: is_class(row["religion"], 3), axis=1)
    ds["Hindu"] = ds.apply(lambda row: is_class(row["religion"], 4), axis=1)
    ds["Ethnic"] = ds.apply(lambda row: is_class(row["religion"], 5), axis=1)
    ds["Marxist"] = ds.apply(lambda row: is_class(row["religion"], 6), axis=1)
    ds["Others"] = ds.apply(lambda row: is_class(row["religion"], 7), axis=1)


    #ds["landmass"] = ds.apply(lambda row: label(row["landmass"], value_mapping["landmass"]), axis=1)
    #ds["zone"] = ds.apply(lambda row: label(row["zone"], value_mapping["zone"]), axis=1)
    #ds["language"] = ds.apply(lambda row: label(row["language"], value_mapping["language"]), axis=1)
    #ds["religion"] = ds.apply(lambda row: label(row["religion"], value_mapping["religion"]), axis=1)

    #check if property is present, the value may vary from 0 to many
    ds["bars_present"] = ds.apply(lambda row: is_present(row["bars"]), axis=1)
    ds["stripes_present"] = ds.apply(lambda row: is_present(row["stripes"]), axis=1)
    ds["circles_present"] = ds.apply(lambda row: is_present(row["circles"]), axis=1)
    ds["crosses_present"] = ds.apply(lambda row: is_present(row["crosses"]), axis=1)
    ds["saltires_present"] = ds.apply(lambda row: is_present(row["saltires"]), axis=1)
    ds["quarters_present"] = ds.apply(lambda row: is_present(row["quarters"]), axis=1)
    ds["sunstars_present"] = ds.apply(lambda row: is_present(row["sunstars"]), axis=1)

    #mainhue
    colors = ds["mainhue"].unique();
    for color in colors:
        ds["mainhue_"+color] = ds.apply(lambda row: is_equal_str(row["mainhue"], color), axis=1)

    #topleft
    colors = ds["topleft"].unique();
    for color in colors:
        ds["topleft_"+color] = ds.apply(lambda row: is_equal_str(row["mainhue"], color), axis=1)

    #botright
    colors = ds["botright"].unique();
    for color in colors:
        ds["botright_"+color] = ds.apply(lambda row: is_equal_str(row["mainhue"], color), axis=1)
    ds = ds.drop(["name", "mainhue", "topleft", "botright"], axis=1)
    print(ds)
    #enc = OneHotEncoder(handle_unknown="ignore")
    #print(enc.fit(ds["landmass"]))
    ds.to_csv("preprocessed.csv", sep=',', index=False)
    return ds

def main():
    #get_test_data()
    #exit()
    learn_df = pd.read_csv('data/amazon_learn.csv')
    #for column in learn_df.columns:
    #    if column == "ID":
    #        continue
    #    learn_df[str(column)+"present"] = learn_df.apply(lambda row: is_present(row[column]), axis=1)
    #learn_df = learn_df[:602]
    #learn_df = pd.concat([learn_df, learn_df], ignore_index=True)
    learn_df.to_csv("preprocessed.csv", sep=',', index=False)
    exit()
    # Read the data from csv files
    ds = get_data_pure()
    print(ds)
    get_data_preprocessed()
    exit()


if __name__ == "__main__":
    main()
