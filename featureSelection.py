from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import itertools
import numpy as np

def create_feature_crosses(df, target_feature, num_new_features):
    """
    Create new features by crossing the most important features.

    Parameters:
    - df: pandas DataFrame containing the features and target.
    - target_feature: string, the name of the target feature column.
    - num_new_features: int, the number of new features to create.

    Returns:
    - df: pandas DataFrame with the new features added.
    """
    # Ensure the target feature is a column in the DataFrame
    if target_feature not in df.columns:
        raise ValueError(f"The target feature '{target_feature}' is not in the DataFrame")

    # Separate the features from the target
    X = df.drop(target_feature, axis=1)
    y = df[target_feature]

    # Fit a random forest to get feature importances
    forest = RandomForestClassifier()
    forest.fit(X, y)

    # Get feature importances
    importances = forest.feature_importances_

    # Convert the importances into a DataFrame
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    # Sort the DataFrame to find the most important features
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Calculate how many top features to consider for creating crosses
    # We need to find the smallest n such that n*(n-1)/2 >= num_new_features
    n = 1
    while n * (n - 1) / 2 < num_new_features:
        n += 1

    # Get the top important features
    top_features = feature_importance_df['Feature'].iloc[:n].values

    # Create new features by crossing the top important features
    # Use itertools.combinations to create unique pairs
    new_features_combinations = list(itertools.combinations(top_features, 2))

    # Now create the new features
    for i, (feature_a, feature_b) in enumerate(new_features_combinations, 1):
        new_feature_name = f'magic_{i}'
        df[new_feature_name] = df[feature_a] * df[feature_b]
        if i == num_new_features:  # Stop after creating the desired number of new features
            break

    return df
