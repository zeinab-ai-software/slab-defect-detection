import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import Lasso
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


def split_data(X, y):  
    X_train, X_combined, y_train, y_combined = train_test_split(X, y, test_size=0.3, random_state=42)  
    X_test, X_val, y_test, y_val = train_test_split(X_combined, y_combined, test_size=0.5)  
    return X_train, X_test, X_val, y_train, y_test, y_val

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

if __name__ == '__main__':
    df_total=pd.read_csv("./Total_Table_Slabs.csv")
    df_labels=pd.read_csv("./SlabLabels.csv")

    X_train, X_test, X_val, y_train, y_test, y_val = split_data(df_total, df_labels)

    feature_names = df_total.columns.tolist()

    model = MultiLabelClassifier(n_estimators=100, random_state=42)

    model.fit(X_train.values, y_train.values, feature_names)

    accuracy, fmeasure = model.evaluate(X_val.values, y_val.values)
   
    print(f"Prediction Accuracy: {accuracy}")
    print(f"Prediction F-measure: {fmeasure}")