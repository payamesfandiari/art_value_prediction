Arthena Data Science Challenge
==============================

This is the Data Science challenge for Arthena

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
# Data Science Internship - Interview Task

## Introduction
Thank you for your interest in Arthena! We were impressed by your application and are happy to consider you for a data science internship. 

We want to get a better sense of how you approach data science problems. Please complete the following task and share your code with us. 

We will review your approach and, if we believe there is a good fit, will reach out to schedule a follow-up call. We aim to respond within seven days and often much sooner.

Applicants usually spend ~2 hours tackling these tasks. This is not designed to be completed in two hours. Again, **you are not expected to finish everything**. We expect you to complete the primary task and write answers to a couple of the open-ended questions. 

## Software Engineering Practices

Follow these software engineering practices:
- Comment appropriately. 
- Commit frequently.
- Only push functional code. 
- Follow a style guide.
- Use a seeded number generator where possible. All functions should be deterministic; I should run your code and get exactly the same results you do.
- Use pipenv (or pipreqs and virtualenv) to create and maintain a virtual environment and requirements list.
- Create a .gitignore (github maintains a Python gitignore [here](https://github.com/github/gitignore/blob/master/Python.gitignore)).

## Instructions

Do all of your work in a Python 3 Jupyter Notebook. Use whatever libraries you want. Follow the aforementioned software engineering practices. Your final deliverable is a "model.py" file that contains an importable "predict" function (see Step 4). 

1. Create a private git repository. Upload the unzipped interview task files and create all of your files in this repository. 
2. Explore the data ("data.csv"). You have auction results for works by 119 different artists. We randomly withheld a portion of auction results by those artists that we will use to test your model. Our test set will contain the same variables but may contain missing values. **The text is encoded in Latin-1 and must be opened using a line like `df = pd.read_csv("data.csv", encoding="latin-1")`**.
3. Prepare and analyze the data. Your primary task is to train a machine learning model that predicts the price of a work of art given its 19 variables. Your target variable is `hammer_price`. Use the root mean squared error as your metric. 
4. Refactor your code into the final deliverable: a file called "model.py" that contains an importable Python function called "predict" that takes in a test CSV file (with the same variables as the data.csv file and an arbitrary number of rows), predicts the price of each row, and returns the RMSE of the predictions. Our test file may contain missing values. Document your solution.
5. Answer a couple of the following questions in a Markdown cell of the Jupyter notebook. Write a couple of sentences per question.
- Which features are most important for your model? Are there any that surprised to you? 
- How would you quantify the uncertainty and/or confidence intervals in the predictions? 
- How would you predict the price of a work if you were only given the artist name, type of work, and size?
- What happened to works by Sol Lewitt before, during and after the 2007-2008 financial crisis. How quickly did prices return to pre-2008 levels?  
- What additional data or features do you think would help increase performance?
- How would you determine the relationship between the size of works and their price?
- How would you make sure the works you're purchasing have uncorrelated returns (so that you can maintain a diverse portfolio)?
- What category of work do you recommend purchasing? 

We designed the task to be completed without additional instructions, but if you spot a mistake or have any questions, please don't hesitate to let us know. 

## Evaluation

We want to learn more about how you use machine learning, statistics, and software engineering to solve problems. We're looking for a demonstration of your ability to understand the problem and understand, prepare, and analyze the data. We care more about how you approach the problem and how you communicate that approach than about how good your numbers are. This is a particularly traditional machine learning task; most of our problems are often much more open-ended and require careful thought to solve. 

We're also looking to evaluate how comfortable you are with software engineering tools and practices like style guides, version control, and commenting. It's okay if you're not used to churning out production code, but you will be working in a team and on a product where mistakes can translate into millions of dollars of losses. We care about the long-term maintanability and extensibility of our code and are happy to help you learn software engineering best practices.

## Authors
* Michael D'Angelo (michael@arthena.com)
* Paul Warren (paul@arthena.com)


