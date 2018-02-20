import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Transforms the given raw Pandas Dataframe into the feature vector necessary for the model.
    The input data must adhere to the format of one artist per row. It should also have the 19 features.

    """

    def __init__(self, categorical_features=True,
                 one_hot_encode=False,
                 gbp__usd=1.41,
                 eur__usd=1.25,
                 top_valid_locs=10,
                 top_valid_materials=None
                 ):
        self.top_valid_locs = top_valid_locs
        self.top_valid_materials = top_valid_materials
        self.return_categorical_features = categorical_features
        self.return_one_hot_encode = one_hot_encode
        self.GBP_USD = gbp__usd
        self.EUR_USD = eur__usd
        self.transformed_features = None

    def fit(self, X, y=None):
        valid_materials = ['paper', 'sculpture', 'canvas', 'prints']
        valid_locs = []

        X.loc[:, 'materials'] = X.materials.apply(lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        X.loc[:, 'category'] = X.category.apply(lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        X.loc[:, 'location'] = X.location.apply(lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        X.loc[:, 'artist_nationality'] = X.artist_nationality.apply(
            lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        if isinstance(self.top_valid_materials, int):
            if self.top_valid_materials < len(X.materials.unique()):
                valid_materials = list(X.materials.value_counts()[:self.top_valid_materials].index)
            else:
                valid_materials = list(X.materials.unique())
        if isinstance(self.top_valid_materials, float):
            last_ind = len(X.materials.unique()) * self.top_valid_materials
            valid_materials = list(X.materials.value_counts()[:last_ind].index)

        if isinstance(self.top_valid_locs, int):
            if self.top_valid_locs < len(X.location.unique()):
                valid_locs = list(X.location.value_counts()[:self.top_valid_locs].index)
            else:
                valid_locs = list(X.location.unique())
        if isinstance(self.top_valid_locs, float):
            last_ind = len(X.location.unique()) * self.top_valid_locs
            valid_locs = list(X.location.value_counts()[:last_ind].index)

        self.valid_features_ = list(X.columns)
        self.valid_materials_ = valid_materials
        self.valid_locs_ = valid_locs

        X.loc[:, 'materials'] = X.materials.apply(self._clean_materials, axis=1)
        X.loc[:, 'location'] = X.location.apply(self._clean_location)

        self.materials_mapping_ = {v: u for u, v in enumerate(self.valid_materials_)}
        self.locations_mapping_ = {v: u for u, v in enumerate(self.valid_locs_)}
        self.category_mapping_ = {v: u for u, v in enumerate(X.category.unique())}
        self.artist_name_mapping_ = {v: u for u, v in enumerate(X.artist_name.unique())}
        self.artist_nationality_mapping_ = {v: u for u, v in enumerate(X.artist_nationality.unique())}

        return self

    def transform(self, X):
        """
        Transforming the data into usable shape.
        :param X: The Pandas dataframe
        :return:
        """
        check_is_fitted(self, 'valid_features_')
        if not isinstance(X, pd.DataFrame):
            raise NotImplementedError("We do not support this type of data. We need Pandas Dataframe as an input.")

        _X = X.drop_duplicates()
        # If it's the train data and we have 'hammer_price' then clean it up
        if "hammer_price" in _X.columns:
            # drop instances where hammer_price is NaN or smaller than zero and there is no estimte of high or low
            _X = _X.loc[~np.logical_and(
                np.logical_and(np.logical_or(_X.hammer_price.isnull(), _X.hammer_price < 0), _X.estimate_high.isnull()),
                _X.estimate_low.isnull())]
            # Replace the negative and NaN hammer_price with a mean of estimate_high and estimate_low
            replacement_neg_hammer_price_ind = np.logical_and(
                np.logical_or(_X.hammer_price < 0, _X.hammer_price.isnull()),
                np.logical_not(_X.estimate_high.isnull()))
            replacement_neg_hammer_price = _X.loc[
                replacement_neg_hammer_price_ind, ['estimate_high', 'estimate_low']].mean(axis=1)
            _X.loc[replacement_neg_hammer_price_ind, 'hammer_price'] = replacement_neg_hammer_price
            # Change all the currencies to USD
            _X.loc[_X.currency == 'GBP', 'hammer_price'] = _X.loc[_X.currency == 'GBP', 'hammer_price'].apply(
                lambda x: x * self.GBP_USD)
            _X.loc[_X.currency == 'EUR', 'hammer_price'] = _X.loc[_X.currency == 'EUR', 'hammer_price'].apply(
                lambda x: x * self.EUR_USD)

        # Making sure strings are all strings !
        _X.loc[:, 'category'] = _X.category.apply(lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        _X.loc[:, 'materials'] = _X.materials.apply(lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        _X.loc[:, 'location'] = _X.location.apply(lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        _X.loc[:, 'artist_nationality'] = _X.artist_nationality.apply(
            lambda x: str(x).lower().replace('\r', '').replace('\n', ' '))

        # Unifying the category feature.
        _X.loc[_X.category == "other works on paper", "category"] = "painting"
        _X.loc[
            np.logical_and(_X.materials == "oil on canvas", _X.category == "unclassified"), 'category'] = "painting"

        _X.loc[np.logical_and(_X.materials == "works on paper",
                              _X.category == "unclassified"), "category"] = "painting"

        _X.loc[np.logical_and(_X.materials == "oil and charcoal",
                              _X.category == "unclassified"), "category"] = "painting"

        _X.loc[
            np.logical_and(_X.materials == "sculpture", _X.category == "unclassified"), "category"] = "sculpture"

        _X.loc[:, 'materials'] = _X.materials.apply(self._clean_materials, axis=1)
        _X.loc[:, 'location'] = _X.location.apply(self._clean_location)
        _X.loc[~_X.location.isin(self.valid_locs_), "location"] = "other"
        _X.loc[_X.location == "nan", "location"] = "other"

        # changing "auction_date" to datetime object
        _X.loc[:, 'auction_date'] = pd.to_datetime(_X.auction_date)
        # Adding new features (year, month,day,week)
        _X = _X.assign(year=[x.year for x in _X.auction_date], month=[x.month for x in _X.auction_date],
                       day=[x.day for x in _X.auction_date], week=[x.week for x in _X.auction_date])

        _X = _X.assign(
            surface=_X.loc[:, ['measurement_width_cm', 'measurement_height_cm', 'measurement_depth_cm']].apply(
                self.surface_volume, axis=1))

        _X = _X.assign(is_artist_dead=(~_X.artist_death_year.isnull()).astype(np.int))
        _X = _X.assign(
            aspect_ratio=_X.loc[:, ["measurement_width_cm", "measurement_height_cm"]].apply(self.calc_aspect_ratio,
                                                                                            axis=1))
        # how long ago this got sold
        current_year = 2018
        _X = _X.assign(years_sold=current_year - _X.year)

        # Number of years since the artist was born
        _X = _X.assign(years_sold=current_year - _X.artist_birth_year)

        dropped_attr = ['artist_birth_year',
                        'artist_death_year',
                        'edition',
                        'auction_date',
                        'year_of_execution',
                        'title',
                        'currency',
                        'estimate_high',
                        'estimate_low'
                        ]
        _X = _X.drop(dropped_attr, axis=1)
        _X = _X.drop_duplicates()
        if self.return_categorical_features:
            _X.loc[:, "materials"] = _X.materials.apply(lambda x: self.materials_mapping_.get(x, -1))
            _X.loc[:, "location"] = _X.location.apply(lambda x: self.locations_mapping_.get(x, -1))
            _X.loc[:, "category"] = _X.category.apply(lambda x: self.category_mapping_.get(x, -1))
            _X.loc[:, "artist_nationality"] = _X.artist_nationality.apply(
                lambda x: self.artist_nationality_mapping_.get(x, -1))
            _X.loc[:, "artist_name"] = _X.artist_name.apply(lambda x: self.artist_name_mapping_.get(x, -1))
            return _X
        if self.return_one_hot_encode:
            _X = pd.get_dummies(_X, columns=["materials", "location", "category", "artist_nationality", "artist_name"])
            return _X

        return _X

    def _clean_materials(self, x, axis=1):
        for m in self.valid_materials_:
            if m in x:
                return m
        return 'other'

    @staticmethod
    def _clean_location(x):
        x = str(x).lower()
        if "," in x:
            return x.split(',')[-1].strip(" ")
        return x.strip(" ")

    @staticmethod
    def surface_volume(x):
        if x.measurement_depth_cm == 0:
            return x.measurement_width_cm * x.measurement_height_cm
        else:
            return x.measurement_width_cm * x.measurement_height_cm * x.measurement_depth_cm

    @staticmethod
    def calc_aspect_ratio(x):
        if x.measurement_width_cm > 0 and x.measurement_height_cm > 0:
            return x.measurement_width_cm / x.measurement_height_cm
        return 0


def main():
    data = pd.read_csv("../../data/raw/data.csv", encoding='latin-1')
    clf = FeatureGenerator()
    clf.fit(data)
    XX = clf.transform(data)
    # Saving the data
    XX.to_csv('../../data/processed/data.csv', encoding='latin-1')
    # Saving the model
    joblib.dump(clf, '../../models/transformer.pkl')


if __name__ == '__main__':
    main()
