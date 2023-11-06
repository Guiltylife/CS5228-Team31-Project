import numpy as np
import pandas as pd
from numpy.random.mtrand import normal
from sklearn.preprocessing import LabelEncoder


class LikelihoodEncoder:
    def __init__(self, cat_feature, target_feature, alpha=1, noise_std=0.1):
        self.cat_feature = cat_feature
        self.target_feature = target_feature
        self.alpha = alpha
        self.noise_std = noise_std
        self.mode_values = None
        self.prior_prob = None

    def fit(self, df):
        # Calculate the prior probabilities (overall distribution of target_feature)
        self.prior_prob = df[self.target_feature].value_counts(normalize=True).to_dict()

        # Calculate the mode for each group with smoothing
        mode_values = df.groupby(self.cat_feature)[self.target_feature].apply(
            lambda x: (x.value_counts() + self.alpha) / (len(x) + self.alpha * len(self.prior_prob)))

        # Replace NaN values with prior probabilities
        mode_values = mode_values.unstack().fillna(self.prior_prob)

        # Get the value with the highest likelihood after smoothing
        self.mode_values = mode_values.idxmax(axis=1)

    def transform(self, df):
        # Map the mode values to the dataframe
        df[f'P({self.target_feature}|{self.cat_feature})'] = df[self.cat_feature].map(self.mode_values)

        # Handle unseen categories by filling NaN with the most common target feature value (or another statistic)
        if self.prior_prob is not None:
            most_common_target_value = max(self.prior_prob, key=self.prior_prob.get)
            df[f'P({self.target_feature}|{self.cat_feature})'].fillna(most_common_target_value, inplace=True)

        # Add Gaussian noise
        df[f'P({self.target_feature}|{self.cat_feature})'] += np.random.normal(0, self.noise_std, size=df.shape[0])

        return df


class GroupedStatsEncoder:
    def __init__(self, target_column, group_columns):
        self.target_column = target_column
        self.group_columns = group_columns
        self.stats_data = {}

    def compute_grouped_stats(self, df, group_column):
        grouped = df[[self.target_column, group_column]].groupby(group_column)

        stats_functions = {
            'std': 'std',
            'median': 'median',
        }

        stats_list = []
        for stat_name, func in stats_functions.items():
            stat_df = grouped.agg(func).reset_index()
            stat_df.columns = [group_column, f"{group_column}_{stat_name}_{self.target_column}"]
            stats_list.append(stat_df)

        stats_df = pd.concat(stats_list, axis=1)
        stats_df = stats_df.loc[:,~stats_df.columns.duplicated()]

        return stats_df

    def fit(self, df):
        for group_column in self.group_columns:
            self.stats_data[group_column] = self.compute_grouped_stats(df, group_column)

    def transform(self, df):
        for group_column in self.group_columns:
            # Merge with the statistics dataframe
            df = df.merge(self.stats_data[group_column], on=group_column, how='left')

            # Fill NaN values with the median of the column
            for stat in ['std', 'median']:
                stat_col_name = f"{group_column}_{stat}_{self.target_column}"
                if stat_col_name in df.columns:
                    df[stat_col_name].fillna(df[stat_col_name].median(), inplace=True)
        return df


def get_economic_indicator_feature(dataset, index_name, value):
    # Convert the "date" column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'])

    # Extract year and month
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month

    # Group by year and month, then calculate mean and standard deviation w.r.t. the value
    monthly_stats = dataset.groupby(['year', 'month'])[value].agg(
        index_price = 'mean',
        index_volatility = 'std'
    ).reset_index()

    return monthly_stats


def encode_hierarchical(train_data, test_data, urban_planning):
    planning_area_col, subzone_col, town_col = urban_planning

    # Initialize the label encoders for each categorical feature
    le_planning_area = LabelEncoder()
    le_subzone = LabelEncoder()
    le_town = LabelEncoder()

    # Fit the label encoders to the categorical features of the training data
    le_planning_area.fit(train_data[planning_area_col])
    le_subzone.fit(train_data[subzone_col])
    le_town.fit(train_data[town_col])

    # Transform both training and testing data, adding 1 to start encoding from 1
    train_data['PlanningAreaCode'] = (le_planning_area.transform(train_data[planning_area_col]) + 1) * 10**5
    test_data['PlanningAreaCode'] = (le_planning_area.transform(test_data[planning_area_col]) + 1) * 10**5

    # For subzone, we reset the encoder for each planning area
    for area in le_planning_area.classes_:
        area_mask = train_data[planning_area_col] == area
        le_subzone.fit(train_data.loc[area_mask, subzone_col])
        train_data.loc[area_mask, 'SubzoneCode'] = (le_subzone.transform(train_data.loc[area_mask, subzone_col]) + 1) * 10**3

        area_mask_test = test_data[planning_area_col] == area
        test_data.loc[area_mask_test, 'SubzoneCode'] = (le_subzone.transform(test_data.loc[area_mask_test, subzone_col]) + 1) * 10**3

    # For town, we reset the encoder for each subzone
    for subzone in train_data[subzone_col].unique():
        subzone_mask = train_data[subzone_col] == subzone
        le_town.fit(train_data.loc[subzone_mask, town_col])
        train_data.loc[subzone_mask, 'TownCode'] = le_town.transform(train_data.loc[subzone_mask, town_col]) + 1

        subzone_mask_test = test_data[subzone_col] == subzone
        test_data.loc[subzone_mask_test, 'TownCode'] = le_town.transform(test_data.loc[subzone_mask_test, town_col]) + 1

    # Create a unique identifier by combining the codes
    train_data['city_encoder'] = train_data['PlanningAreaCode'] + train_data['SubzoneCode'] + train_data['TownCode']
    test_data['city_encoder'] = test_data['PlanningAreaCode'] + test_data['SubzoneCode'] + test_data['TownCode']

    # Drop the intermediate columns used for encoding
    train_data.drop(['PlanningAreaCode', 'SubzoneCode', 'TownCode'], axis=1, inplace=True)
    test_data.drop(['PlanningAreaCode', 'SubzoneCode', 'TownCode'], axis=1, inplace=True)

    return train_data, test_data