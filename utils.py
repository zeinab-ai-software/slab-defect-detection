def split_data(X, y):  
    X_train, X_combined, y_train, y_combined = train_test_split(X, y, test_size=0.3, random_state=42)  
    X_test, X_val, y_test, y_val = train_test_split(X_combined, y_combined, test_size=0.5)  
    return X_train, X_test, X_val, y_train, y_test, y_val
