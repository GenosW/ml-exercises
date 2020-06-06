# Exercise 2: Classification

Next meeting: **28. - 31.5.2020**

## Description

**Submission deadline: 9.06.2020, 23:59** (might change)

**Presentation: 18.06.2020,  13:00 - 15:00, Zoom**

Group#: 19

Techniques:

    a) kNN
    b) RandomForest
    c) Multilayer Perceptron: https://scikit-learn.org/stable/modules/neural_networks_supervised.html#classification

You need to chose a total of 4 datasets.

    - 2 by ourselves (from exercise 0 or new ones)
    - 1 from small datasets below (depen)
    - 1 large dataset

Your submission should contain:

    - Your written report
    - All your scripts, etc. that you created. If you used only a GUI and didn't create any software artefacts, please state that in your report
    - All the datasets that you used (except the ones from Kaggle)

## Work flow

    - Get your data sets:
        – Your existing ones & from Kaggle
    - Import data file, scale/encode data, other preprocessing
    - Run classifiers (with different parameters)
        – Select most interesting ones, ...
        – Document any problems/findings
        – Upload results from good algorithms to Kaggle
        – Not necessary to implement algorithms
        - Rely on libraries, modules etc.
        - Code just for loading data, pre-processing,runningconfigurations, processing/aggregating results, ...
    - Write your report

## Dataset 1: Caravana (Ok/Nok)

DS: Aleks

## Dataset 2: Wine Quality (Poor/Normal/Excellent)

DS: Alex

https://archive.ics.uci.edu/ml/datasets/Wine+Quality

Plan B: Zoo (/Mushroom)

## Dataset 3: Breast Cancer (Small) (Ok/Nok)

DS: Peter

https://www.kaggle.com/c/184702-tu-ml-ws-19-breast-cancer/data

A nice kernel from Kaggle (good visualization): https://www.kaggle.com/vincentlugat/breast-cancer-analysis-and-prediction/comments

Another one: https://www.kaggle.com/buddhiniw/breast-cancer-prediction#Using-the-Wisconsin-breast-cancer-diagnostic-data-set-for-predictive-analysis

## Dataset 4: Amazon Review Data (Large)

First everyone here:

    - kNN: Aleks 
    - RandomForest: Peter 
    - MlP: Alex

https://www.kaggle.com/c/184702-tu-ml-ws-19-amazon-commerce-reviews

Our Group: SS20 - Group19

## Kaggle: How-To

For the classification exercise, we utilise Kaggle inClass as an evaluation platform, which will provide you immediate feedback on how well your solutions are compared to the other students in the course. Here are some details about this competition:

    Kaggle is a platform that allows a competition for a certain data set. Participants submit their prediction on a test set, and will get automated scoring on their results, and will enter the leaderboard.
    From Kaggle, you will be able to obtain a labelled training set, and an unlabelled test set.
    You can submit multiple entries to Kaggle; for each entry, you need to provide details on how you achieved the results: which software and which version of the software, which operating system, which algorithms, and which parameter settings for these algorithms; further, any processing applied to the data before training/predicting. There is a specific "description" field when submitting, you should fill in this information there, and you also need to include this description and the actual submission file in your final submission to TUWEL.
    To submit to Kaggle, you need to create a specific submission file, which contains the predictions you obtain on the test set. Computing an aggregated evaluation criterion is done automatically by Kaggle
    The format of your submission is rather simple - it is a comma-separated file, where the first column is the identifier of the item that you are predicting, and the second column is the class you are predicting for that item. The first line should include a header, and is should use the names provided in the training set. An example is below:

    ID,class
    911366,B
    852781,B
    89524,B
    857438,B
    905686,B

    For the top teams submitting to Kaggel, you will be able to obtain up to 10% of your final points for this exercise as bonus reward.
    You can form teams in Kaggle, to submit together with your course group members. Please name your group according to your group in TUWEL, e.g. "Group01", to allow us to map you to TUWEL.
    There is a limit of 7 submissions per day; start early enough to try various settings. Finally, you also need to select your top 7 submissions to be counted in the competition
    Evaluation in Kaggle is split in two types of leaderboards - the private and public one. Here, the data is split into roughly 50% / 50%, and as soon as you upload, you will know your results on one of these splits. The final results will only be visible once the competition closes, and as it is computed on the other split, might be slightly different than what you see initially (e.g. this is similar to a training/test/validation split)
    You must not use the ID attributes that are part of the dataset. These are just for you to be able to generate a mapping (ID, prediction), but they must not be used as input for the machine learning model! 


---
