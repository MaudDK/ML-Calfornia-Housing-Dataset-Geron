
import pandas as pd
import numpy as np
from joblib import parallel_backend
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
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

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.add_rooms_per_bedrooms:
            rooms_per_bedrooms = X[:, rooms_ix] / X[:, bedrooms_ix]
            X = np.c_[X, rooms_per_bedrooms]

        if self.add_households_per_population:
            households_per_population = X[:,
                                          households_ix] / X[:, population_ix]
            X = np.c_[X, households_per_population]

        if self.add_households_per_total_rooms:
            households_per_rooms = X[:, households_ix] / X[:, rooms_ix]
            X = np.c_[X, households_per_rooms]

        return X

# Top Feature Selector


def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(
            self.feature_importances, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices_]

# Display Score Method


def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Standard Deviation:', scores.std())


# Build Pipeline
encodes = list(housing['ocean_proximity'].value_counts().index)
num_attribs = list(X_train)[: -1]
extra_attribs = ['rooms_per_bedrooms',
                 'households_per_population', 'households_per_rooms']
cat_attribs = ['ocean_proximity']

rf_reg = RandomForestRegressor(random_state=42)

num_transformer = Pipeline([('imputer', SimpleImputer(strategy="median")),
                            ('attribs_adder', CombinedAttributesAdder()),
                            ('std_scaler', StandardScaler()),
                            ])


preprocessor = ColumnTransformer([('num', num_transformer, num_attribs),
                                  ('cat', OneHotEncoder(), cat_attribs)])


# Transform Data
X_train_prep = preprocessor.fit_transform(X_train)


# Grid Search
param_grid = [
    {'bootstrap': [False], 'max_depth': [32], 'max_features': [
        5], 'n_estimators': [700], 'min_samples_leaf': [3], 'min_samples_split': [8]},
    {'bootstrap': [False], 'max_depth': [32], 'max_features': [
        5], 'min_samples_leaf': [3], 'min_samples_split': [8], 'n_estimators': [700]},
    {'bootstrap': [False], 'max_depth': [16],
     'max_features': [6], 'n_estimators': [400]},
    {'bootstrap': [False], 'max_depth': [30],
     'max_features': [6], 'n_estimators': [500]},
    {'bootstrap': [False], 'max_depth': [32],
     'max_features': [5], 'n_estimators': [600]},
    {'bootstrap': [False], 'max_depth': [32],
     'max_features': [5], 'n_estimators': [700]},
    {'n_estimators': [900], 'min_samples_split': [3], 'min_samples_leaf': [1], 'max_features': [5], 'max_depth': [45], 'bootstrap': [False]}]

with parallel_backend('threading', n_jobs=5):
    grid_search = GridSearchCV(rf_reg, param_grid, cv=2, scoring='neg_mean_squared_error',
                               return_train_score=True, verbose=10, n_jobs=5)
    grid_search.fit(X_train_prep, y_train)

feature_importances = grid_search.best_estimator_.feature_importances_
cat_encoder = preprocessor.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances, attributes), reverse=True))

final_model = grid_search.best_estimator_

X_test_prepared = preprocessor.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

full_pipeline = Pipeline([('prep', preprocessor), ('feature_selector',
                                                   TopFeatureSelector(feature_importances, 10)), ('clf', final_model)])

full_pipeline.fit(X_train, y_train)
final_predictions = full_pipeline.predict(X_test)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
