num_transformer = Pipeline([('imputer', SimpleImputer(strategy="median")),
                            ('attribs_adder', CombinedAttributesAdder()),
                            ('std_scaler', StandardScaler()),
                            ])

preprocessor = ColumnTransformer([('num', num_transformer, num_attribs),
                                  ('cat', OneHotEncoder(), cat_attribs)])


X_train = preprocessor.fit_transform(X_train)