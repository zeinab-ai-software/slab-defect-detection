import pandas as pd
from src.model import MultiLabelClassifier
from src.utils import split_data

if __name__ == '__main__':
    # Load data
    df_total = pd.read_csv("./data/Total_Table_Slabs.csv")
    df_labels = pd.read_csv("./data/SlabLabels.csv")

    # Split data
    X_train, X_test, X_val, y_train, y_test, y_val = split_data(df_total, df_labels)
    feature_names = df_total.columns.tolist()

    # Train model
    model = MultiLabelClassifier(n_estimators=100, random_state=42)
    model.fit(X_train.values, y_train.values, feature_names)

    # Evaluate on validation set
    accuracy, fmeasure = model.evaluate(X_val.values, y_val.values)

    print("===== Evaluation Results =====")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation F1 Score: {fmeasure:.4f}")
