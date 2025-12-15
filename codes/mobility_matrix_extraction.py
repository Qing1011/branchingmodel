import pandas as pd
import numpy as np
import os
import csv
import datetime as d
from datetime import datetime as D
import gzip
import sys
import datetime
import ast


def process_visitor_data(visitor_home_list, raw_visits, my_fips):
    """
    Filters out 'CA:' entries, normalizes visitor numbers, and maps the home locations 
    to the indices of the 'my_fips' dataframe.

    Args:
    - visitor_home_list (list): List of strings formatted as '<location>:<number>'.
    - my_fips (pd.DataFrame): DataFrame containing FIPS data with an 'FIPS' column.

    Returns:
    - pd.Series: A series containing the indices in 'my_fips' for the home locations 
                 and the normalized visitor counts.
    """

    # Filter out 'CA:' entries
    filtered_home_list = [
        item for item in visitor_home_list if not item.startswith('CA:')]

    # Extract the home locations and visitor numbers
    visitor_cbg_list = [str(i.split(':')[0]) for i in filtered_home_list]
    visitor_num_array = np.array([int(i.split(':')[1])
                                 for i in filtered_home_list])

    # Normalize visitor numbers
    total_visitor_num = np.sum(visitor_num_array)
    weight = visitor_num_array / total_visitor_num
    loss = raw_visits - total_visitor_num

    normalized_visitor_array = visitor_num_array + loss * weight

    # Map home locations to indices in 'my_fips'
    home_indices = []
    modified_visitor_counts = []

    for home, count in zip(visitor_cbg_list, normalized_visitor_array):
        if len(home) > 6:  # Ensure valid home location format
            home_county = np.int64(home[:5])
            filtered_fips = my_fips[my_fips['FIPS'] == home_county]
            if not filtered_fips.empty:
                home_indices.append(filtered_fips.index[0])
                modified_visitor_counts.append(count)

    return pd.Series([home_indices, modified_visitor_counts])


def dayvisits_by_visitor(modified_visit_pervisitor, daily_visitor_weight):
    """
    Converts the number of visitors per day to the number of visitors per day of the week.
    find [fips, the visits by day (a list of j:list of 7 length)]
    """
    vis_daily_matrix = []  # a list of j:list of 7 length,

    try:
        for num in modified_visit_pervisitor:
            num_week = num*daily_visitor_weight
            vis_daily_matrix.append(num_week)
    except:
        print(modified_visit_pervisitor)
    return vis_daily_matrix


def mobility_extract_per_poi(df_j, M_raw):
    """
    Extracts the mobility matrix for a single POI.
    """
    home_idx_list = df_j['home_idx_list']
    poi_idx = np.int64(df_j['poi_idx'])
    dayvisit_list = df_j['vis_daily_matrix']
    for v, home_idx in enumerate(home_idx_list):
        vis_daily_a = dayvisit_list[v]
#         print(vis_daily_a)
        M_raw[:, poi_idx, home_idx] = vis_daily_a + M_raw[:, poi_idx, home_idx]
    return M_raw
