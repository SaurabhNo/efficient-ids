import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def correlation_feature_selection(file, threshold=0.1):
    # Read dataset
    df = pd.read_csv(file)

    # Find label column
    label_column = None

    possible_labels = [
        "Label",
        "label",
        "Target",
        "target",
        "Class",
        "class"
    ]

    for column in possible_labels:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError("Label column not found in dataset")

    # Separate features and label
    X = df.drop(columns=[label_column])
    y = df[label_column]

    # Keep only numeric features
    X_numeric = X.select_dtypes(include="number")

    if X_numeric.empty:
        raise ValueError("No numeric features found")

    # Convert label into numeric values
    y_numeric = pd.factorize(y)[0]

    # Calculate correlation with target
    correlations = {}

    for column in X_numeric.columns:
        correlation = X_numeric[column].corr(
            pd.Series(y_numeric)
        )

        if pd.notna(correlation):
            correlations[column] = round(
                abs(float(correlation)), 4
            )
        else:
            correlations[column] = 0.0

    # Sort features by correlation
    sorted_features = sorted(
        correlations.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Select features above threshold
    selected_features = [
        feature
        for feature, score in sorted_features
        if score >= threshold
    ]

    return {
        "method": "Correlation",
        "label_column": label_column,
        "threshold": threshold,
        "total_numeric_features": len(X_numeric.columns),
        "selected_feature_count": len(selected_features),
        "selected_features": selected_features,
        "feature_scores": correlations
    }

def mutual_information_feature_selection(file, top_k=10):
    # Read dataset
    df = pd.read_csv(file)

    # Find label column
    label_column = None

    possible_labels = [
        "Label",
        "label",
        "Target",
        "target",
        "Class",
        "class"
    ]

    for column in possible_labels:
        if column in df.columns:
            label_column = column
            break

    if label_column is None:
        raise ValueError("Label column not found in dataset")

    # Separate features and target
    X = df.drop(columns=[label_column]).copy()
    y = df[label_column].copy()

    # Convert categorical feature columns to numeric
    for column in X.columns:
        if X[column].dtype == "object":
            X[column] = pd.factorize(X[column])[0]

    # Convert target to numeric
    if y.dtype == "object":
        y = pd.factorize(y)[0]

    # Handle missing values
    X = X.fillna(0)

    # Calculate Mutual Information scores
    scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    # Store scores
    feature_scores = {}

    for feature, score in zip(X.columns, scores):
        feature_scores[feature] = round(float(score), 4)

    # Sort features by score
    sorted_features = sorted(
        feature_scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    # Select top K features
    selected_features = [
        feature
        for feature, score in sorted_features[:top_k]
    ]

    return {
        "method": "Mutual Information",
        "label_column": label_column,
        "total_features": len(X.columns),
        "selected_feature_count": len(selected_features),
        "top_k": top_k,
        "selected_features": selected_features,
        "feature_scores": feature_scores
    }