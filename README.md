Arthena Data Science Challenge
==============================

This is the Data Science challenge for Arthena. The project follows the Datascience template in Cookiecutter. 

## Project details
First I started by looking at different attributes *(Note : I use attribute,feature,and KPI interchangeably)* and try to understand what exactly was happening in the data.
The first step was to design the Feature Transformer to transform the raw data into a shape usable by Machine Learning algorithms. 
`FeatureTransformer` in `src.features.build_features` module contains the logic of the transformation. You can also see the same steps in the Jupyter notebook named `3.1-pe-more-feature-eng`.

I have engineered several new features based on the existing features. The detail of these features are listed below : 
1. `year   : The year of the Auction`
2. `month  : The month of the Auction` 
3. `day  : The day of the Auction`
4. `week  : The week of the year of the Auction`
5. `surface : Width * Height OR Width * Height * Depth`
6. `years_sold : How long ago this piece was sold`
7. `is_artist_dead : Is the artist dead?`
8. `aspect_ratio : Width / Height`


After extracting all the features, I started testing different learning methods. Some of the tests are included in the notebooks 
`3.0-pe-model-selection` and `2.0-pe-feature-engineering`. I found out that Linear Models like Ridge Regression or Lasso does not perform particularly well.
For this I quickly started looking at Tree-based methods like Decision Tree based models and Gradient Trees. Among them, 
AdaBoost Regression model performed well and the best model was Random Forest Regression. 

Instead of using One Hot Encoding techniques to transform the data, since we are dealing with Tree based model, I used factorizing transformation 
where for every unique value in a column, I assigned an integer. This information is saved in the Feature Transformer to be used later.  

## What if I had more time ?
The first thing I do if I had more time is to gather more data specifically about the actual pieces (For example the picture of the painting)
this will help to create a profile (using Convolutional Neural Net) for different art piece. I would also create a model for every artist separately. 
The data did not have any information about the auction itself which could be very helpful. 
Unfortunately, I did not had the time to investigate Neural nets specifically Autoencoders and RNNs. I have a feeling that by tracking each individual art piece as they get purchased 
and later sold, we can create a fine grain profile for each individual piece which will be useful to look at. 

Another important and interesting problem that I encounter was that none of the simpler linear models worked which is usually a bit odd. This means that 
the predictive power of the features was not enough for the model to be able to extract any linear interaction. I think this 
would be something to further investigate and understand.    

## Deliverables 
* `Final Experiments` notebook, contains all the steps I ran to get the trained models. It also contains the answer to a subset of questions below. 

* `model.py` has a importable function called `predict` which will either get a path to an existing csv file or a Pandas dataframe. 


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.        
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
    │   │   └── train_model.py <- Contains the end model
    │ 
    └-- model.py           <- Contains the predict function which will take in a csv file path and returns RMSE  
   

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
