import pandas as pd


def analyze_dataset(file):
    df = pd.read_csv(file)

    # Basic dataset information
    rows = int(df.shape[0])
    columns = int(df.shape[1])

    # Column information
    column_names = df.columns.tolist()

    # Missing values
    missing_values = int(df.isnull().sum().sum())

    # Duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # Numeric and categorical columns
    numeric_columns = int(
        df.select_dtypes(include="number").shape[1]
    )

    categorical_columns = int(
        df.select_dtypes(exclude="number").shape[1]
    )

    # Memory usage in MB
    memory_usage_mb = round(
        df.memory_usage(deep=True).sum() / (1024 * 1024),
        2
    )

    # Try to find the target/label column
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

    # Class distribution
    class_distribution = {}

    if label_column:
        distribution = df[label_column].value_counts()

        for key, value in distribution.items():
            class_distribution[str(key)] = int(value)

    return {
        "rows": rows,
        "columns": columns,
        "column_names": column_names,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "memory_usage_mb": memory_usage_mb,
        "label_column": label_column,
        "class_distribution": class_distribution
    }