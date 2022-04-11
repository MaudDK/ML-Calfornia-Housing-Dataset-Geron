from statistics import LinearRegression
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


housing = pd.read_csv("datasets\housing\housing.csv")


# Stratify Spliting Based on Income Cats in bins
bins = [0., 2.5, 3.5, 4.7, 8, np.inf]
housing['income_cat'] = pd.cut(
    housing["median_income"], bins=bins, labels=range(len(bins)-1))

X = housing.drop('median_house_value', axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=housing['income_cat'])


# Removing Income Cats from dataset as it is no longer needed
for dataset in [X_train, X_test, housing]:
    dataset.drop('income_cat', axis=1, inplace=True)


# Combines Attributes , see the explore notebook to figure out how these attributes were discovered
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


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


num_attribs = list(X_train)[:-1]
cat_attribs = ['ocean_proximity']

num_transformer = Pipeline([('imputer', SimpleImputer(strategy="median")),
                            ('attribs_adder', CombinedAttributesAdder()),
                            ('std_scaler', StandardScaler()),
                            ])

preprocessor = ColumnTransformer([('num', num_transformer, num_attribs),
                                  ('cat', OneHotEncoder(), cat_attribs)])

full_pipeline = Pipeline([('preprocessor', preprocessor),
                          ('classifier', LinearRegression())])


X_train = preprocessor.fit_transform(X_train)
print(X_train)
