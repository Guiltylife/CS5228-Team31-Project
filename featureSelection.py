from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import itertools

def determine_feature_crosses(df, target_feature, num_new_features):
    """
    Determine which feature crosses to create based on feature importances.

    Parameters:
    - df: pandas DataFrame containing the features and target.
    - target_feature: the name of the target feature column.
    - num_new_features: the number of new feature crosses to create.

    Returns:
    - feature_pairs: a list of tuples, where each tuple contains a pair of feature names.
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

    # Create a list of feature pairs to cross
    feature_pairs = list(itertools.combinations(top_features, 2))[:num_new_features]

    return feature_pairs

def create_feature_crosses(df, feature_pairs):
    """
    Create new features by crossing the given pairs of features.

    Parameters:
    - df: pandas DataFrame containing the features.
    - feature_pairs: list of tuples, each tuple contains a pair of features to cross.

    Returns:
    - df: pandas DataFrame with the new features added.
    """
    # Now create the new features
    for i, (feature_a, feature_b) in enumerate(feature_pairs, 1):
        new_feature_name = f'magic_{i}'
        df[new_feature_name] = df[feature_a] * df[feature_b]

    return df
