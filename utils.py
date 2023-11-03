import numpy as np
import pandas as pd
from numpy.random.mtrand import normal


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
        df[f'likelihood_{self.cat_feature}'] = df[self.cat_feature].map(self.mode_values)

        # Add Gaussian noise
        df[f'likelihood_{self.cat_feature}'] += np.random.normal(0, self.noise_std, size=df.shape[0])

        return df


class GroupedStatsEncoder:
    def __init__(self, target_column, group_columns):
        self.target_column = target_column
        self.group_columns = group_columns
        self.stats_data = {}

    def compute_grouped_stats(self, df, group_column):
        grouped = df[[self.target_column, group_column]].groupby(group_column)

        stats_functions = {
            'size': 'size',
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
            df = df.merge(self.stats_data[group_column], on=group_column, how='left')
        return df

def get_economic_indicator_feature(dataset, value):
    # Convert the "date" column to datetime format
    dataset['date'] = pd.to_datetime(dataset['date'])

    # Extract year and month
    dataset['year'] = dataset['date'].dt.year
    dataset['month'] = dataset['date'].dt.month

    # Group by year and month, then calculate mean and standard deviation w.r.t. the value
    monthly_stats = dataset.groupby(['year', 'month'])[value].agg(
        index_price='mean',
        index_volatility='std'
    ).reset_index()

    return monthly_stats