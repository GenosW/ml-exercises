import pandas as pd
import numpy as np


def main():
    o = pd.read_csv("preprocessed.csv")
    n = pd.read_csv("valid.csv")
    x = pd.concat([o, n], ignore_index=True)
    x = x.drop(["Unnamed: 0"], axis=1)
    x.to_csv("new_learning_data.csv", index=False)


main()
