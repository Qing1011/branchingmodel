import pandas as pd
import numpy as np
import os
import csv
import gzip
import sys
import datetime
import ast
from mobility_matrix_extraction import *

ym_datesofweek = np.load('../data/ym_datesofweek.npy',
                         allow_pickle=True).item()
num_files = np.load('../data/num_files.npy', allow_pickle=True).item()
real_data_path = '/Users/qingyao/OneDrive - cumc.columbia.edu/machine_learning_data/'
fips_main = pd.read_csv(
    real_data_path+'county_data/fips_mainland.csv', names=['FIPS'])
fips_main_sorted = fips_main.sort_values(by='FIPS')
fips_main_sorted['poi_idx'] = fips_main_sorted.index
# Assuming necessary imports for the functions `process_visitor_data`, `dayvisits_by_visitor`, and `mobility_extract_per_poi` are present at the top


def extract_mobility_matrix(data_path, save_dir, y, m):
    week_start_dates = ym_datesofweek[y + m]
    csv.field_size_limit(500 * 1024 * 1024)
    for start_day in week_start_dates[:1]:
        # print(f"Processing for start_day: {start_day}")
        # there three files which is not data
        n_file = num_files['/' + y + '/' + m +
                           '/' + start_day + '/SAFEGRAPH/WP'] - 3
        M_w = np.zeros((7, 3108, 3108))
        for i in range(1, n_file+1):
            print(f"Processing file {i}")
            new_name = data_path + y + '_' + m + \
                '_' + start_day + '_{}.csv'.format(i)
            df_file = pd.read_csv(new_name)
            df_noca = df_file[~df_file['poi_cbg'].astype(
                str).str.contains('CA:')]
            df_noca['poi_fips'] = df_noca['poi_cbg'].astype(
                str).str[:-7].astype(int)
            df_i = pd.merge(df_noca, fips_main_sorted,
                            left_on='poi_fips', right_on='FIPS', how='left')
            df_i_main = df_i.dropna()

            df_m = pd.DataFrame()
            df_m[['raw_visit_counts', 'poi_idx']
                 ] = df_i_main[['raw_visit_counts', 'poi_idx']]
            df_m['visits_by_day_list'] = df_i_main['visits_by_day'].apply(
                lambda x: np.array(ast.literal_eval(x)))
            df_m['multiplier'] = df_m.apply(
                lambda x: x['visits_by_day_list']/x['raw_visit_counts'], axis=1)  # multiplier
            df_m['visitor_home_list'] = df_i_main['visitor_home_aggregation'].apply(
                lambda x: x[1:-1].replace('"', '').split(','))
            df_m[['home_idx_list', 'modified_visit_list']] = df_m.apply(lambda x: process_visitor_data(
                x['visitor_home_list'], x['raw_visit_counts'], fips_main_sorted), axis=1)
            df_m['vis_daily_matrix'] = df_m.apply(lambda x: dayvisits_by_visitor(
                x['modified_visit_list'], x['multiplier']), axis=1)
            for j_data_idx in range(len(df_m)):
                df_j = df_m.iloc[j_data_idx]
                M_w = mobility_extract_per_poi(df_j, M_w)

        for d in range(7):
            np.savetxt(save_dir+'M_{}{}_{}.csv'.format(y, m,
                       (int(start_day)+d)), M_w[d], delimiter=",")
    return


if __name__ == '__main__':
    data_path = '../../../../../../../Volumes/Seagate_Qing/Safegraph_clean/'
    save_dir = '/Users/qingyao/Documents/branching_data/'
# np.save('ym_datesofweek',ym_datesofweek)
    y = '2020'
    for m in ['04']:
        extract_mobility_matrix(data_path=data_path,
                                save_dir=save_dir, y=y, m=m)
