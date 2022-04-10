import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from sklearn.base import BaseEstimator, TransformerMixin
# We want [total_rooms per total bedrooms] and [households per population] and [households per total rooms]

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

num_attribs = []
cat_attribs = []


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_rooms_per_bedrooms=True, add_households_per_population=True, add_households_per_total_rooms=True):
        self.add_rooms_per_bedrooms = add_rooms_per_bedrooms
        self.add_households_per_population = add_households_per_population
        self.add_households_per_total_rooms = add_households_per_total_rooms

    def fit(self, X):
        return self

    def transform(self, X):
        transforms = []
        if self.add_rooms_per_bedrooms:
            rooms_per_bedrooms = X[:, rooms_ix] / X[:, bedrooms_ix]
            transforms.append(rooms_per_bedrooms)

        if self.add_households_per_population:
            households_per_population = X[:,
                                          households_ix] / X[:, population_ix]
            transforms.append(households_per_population)

        if self.add_households_per_total_rooms:
            households_per_rooms = X[:, households_ix] / X[:, rooms_ix]
            transforms.append(households_per_rooms)
            return np.c_[X, rooms_per_bedrooms, households_per_population, households_per_rooms]

        else:
            return np.c_[X, rooms_per_bedrooms, households_per_population]


num_attribs = []
cat_attribs = []

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),
                         ])

full_pipeline = ColumnTransformer([('num', num_pipeline, num_attribs),
                                   ('cat', OneHotEncoder(), cat_attribs)])
