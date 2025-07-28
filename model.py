class MultiLabelClassifier:
   
    def __init__(self, n_estimators=100, random_state=42):
        self.feature_selectors = []
        self.selected_features = []
        self.multi_output_clf = MultiOutputClassifier(
            RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        )

    def fit(self, X_train, y_train, feature_names):
        lasso = Lasso()
        multi_output_regressor = MultiOutputRegressor(lasso)
        multi_output_regressor.fit(X_train, y_train)

        for estimator in multi_output_regressor.estimators_:
            feature_selector = SelectFromModel(estimator, prefit=True)
            self.feature_selectors.append(feature_selector)
            self.selected_features.extend(
                [feature_names[i] for i in feature_selector.get_support(indices=True)]
            )
       
        X_train_selected = self.feature_selectors[-1].transform(X_train)
        self.multi_output_clf.fit(X_train_selected, y_train)

    def predict(self, X):
        X_selected = self.feature_selectors[-1].transform(X)
        return self.multi_output_clf.predict(X_selected)

    @staticmethod
    def _accuracy(y_true, y_pred):
        temp = 0
        if y_true.shape[0] == 0:
            return 0
        for i in range(y_true.shape[0]):
            numerator = np.sum(np.logical_and(y_true[i], y_pred[i]))
            denominator = np.sum(np.logical_or(y_true[i], y_pred[i]))
            if denominator > 0:
                temp += numerator / denominator
        return temp / y_true.shape[0]

    @staticmethod
    def _f1_measure(y_true, y_pred):
        temp = 0
        if y_true.shape[0] == 0:
            return 0
        for i in range(y_true.shape[0]):
            if np.sum(y_true[i]) == 0 and np.sum(y_pred[i]) == 0:
                temp += 1.0
                continue
            numerator = 2 * np.sum(np.logical_and(y_true[i], y_pred[i]))
            denominator = np.sum(y_true[i]) + np.sum(y_pred[i])
            if denominator > 0:
                temp += numerator / denominator
         return temp / y_true.shape[0]
   
    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = self._accuracy(y_true, y_pred)
        fmeasure = self._f1_measure(y_true, y_pred)
        return accuracy, fmeasure
