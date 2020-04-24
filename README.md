# ml-exercises
Repo for our machine learning exercises in SS2020.

Authors:

    - Alexander Leitner
    - Aleksander Hadzhiyski
    - Peter Holzner

---

## Exercise 0: Dataset registration & description

Submission deadline: 25.03.2020, 23:59

Regression: [Moneyball](https://www.openml.org/d/41021)

Classification: [Carabana - Don't get kicked](https://www.openml.org/d/41162)

---

## Exercise 1:

### Description

**Submission deadline: 4.05.2020, 23:59**

Choose 4 techniques from:

    a) linear regression
    b) polynominal regression
    c) logarithmic regression
    d) kNN
    e) Lasso
    f) Ridge
    g) Regression tree
    h) ...?

You need to chose a total of 4 datasets.
- 1 from exercise 0
- 1 from Kaggle/UCI ML Repository (or other repositories)
- 2 from list below

Choose 2 from: 

Bias correction of numerical prediction model temperature forecast Data Set
http://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast   
Data Folder: http://archive.ics.uci.edu/ml/machine-learning-databases/00514/

Some info:

    > Number of Instances: 7750
    > Number of Attributes: 25
    > Missing Values: Yes

Verdict: 

    > # of samples: med
    > # of dimensions: med-high

QSAR fish toxicity Data Set
http://archive.ics.uci.edu/ml/datasets/QSAR+fish+toxicity
Data Folder: http://archive.ics.uci.edu/ml/machine-learning-databases/00504/

Some info:

    > Number of Instances: 908
    > Number of Attributes: 7

Verdict: 

    > # of samples: low
    > # of dimensions: low

Metro Interstate Traffic Volume Data Set
http://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
Data Folder:http://archive.ics.uci.edu/ml/machine-learning-databases/00492/

Some info:

    > Number of Instances: 48204
    > Number of Attributes: 9

Verdict: 

    > # of samples: high
    > # of dimensions: low

Real estate valuation data set Data Set
http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
Data Folder: http://archive.ics.uci.edu/ml/machine-learning-databases/00477/

Some info:

    > Number of Instances: 414
    > Number of Attributes: 7

Verdict: 

    > # of samples: low
    > # of dimensions: low

---

### Meetings:

Meeting 1: Fr., 24.04.2020

Next meeting: Monday evening, 17:00

Chosen:

    a) linear regression
    b) kNN
    c) Lasso
    d) random forest --> regression tree?

How:

    > Jupyter notebook
    > bissl plots
    > bissl code zum spielen
    > bissl ergebnisse/fazits
    > bissl pros/cons

Who & what (on Moneyball):

    a) linear regression    --> Alex
    b) kNN                  --> Aleks
    c) Lasso:               --> Code von Alex
    d) random forest        --> Peter

---

### Dataset 1: Moneyball {low/low}

Some info:

    > Number of Instances: 1230
    > Number of Attributes: 15

Verdict: 

    > # of samples: low
    > # of dimensions: low-med

---

### Dataset 2: Video Games Sales (Kaggle) {med/low-med} [?]

[Video Games Sales with Metacritic ratings](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings)

Dimensionality analysis:

    > Rows: 16719
    > Columns: 16
    > Column names: ['Name', 'Platform', 'Year_of_Release', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Developer', 'Rating']

Possible targets:

    > Sales: so either of ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    > Rating: so either of ['Critic_Score', 'User_Score']

---

### Dataset 3: Metro Interstate Traffic Volume Data Set {high/low}

http://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
Data Folder:http://archive.ics.uci.edu/ml/machine-learning-databases/00492/

Some info:

    > Number of Instances: 48204
    > Number of Attributes: 9

Verdict: 

    > # of samples: high
    > # of dimensions: low

<!-- ### Dataset 3.b: Real estate valuation data set Data Set
http://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
Data Folder: http://archive.ics.uci.edu/ml/machine-learning-databases/00477/

Some info:

    > Number of Instances: 414
    > Number of Attributes: 7

Verdict: 

    > # of samples: low
    > # of dimensions: low -->

---

### Dataset 4: Bias correction of numerical prediction model temperature forecast Data Set {med/med}
http://archive.ics.uci.edu/ml/datasets/Bias+correction+of+numerical+prediction+model+temperature+forecast   
Data Folder: http://archive.ics.uci.edu/ml/machine-learning-databases/00514/

Some info:

    > Number of Instances: 7750
    > Number of Attributes: 25

Verdict: 

    > # of samples: med
    > # of dimensions: med-high

---

## Cheat Sheets:

Here is github repo with some cheat sheets for various Data Science/Machine Learning related cheat sheets:

https://github.com/abhat222/Data-Science--Cheat-Sheet

Other cheat sheets are saved in the folder of the same name.
