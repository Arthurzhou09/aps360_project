import pandas as pd
import argparse
import os
import sys
sys.path.append('/Users/arthurzhou/github/aps360_project')
from data.data_utils import load_data, compare_single_fitness



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing the .xlsx files containing the DMS data')
    parser.add_argument('--output_file', type=str, default='/Users/arthurzhou/github/aps360_project/src/data/process/processed_data.csv', help='Path to save the processed data as a CSV file')
    parser.add_argument('--error_threshold', type=float, default=1.0, help='Error threshold for filtering double mutant entries based on single mutant fitness values')
    args = parser.parse_args()

    # processing should load both dfs in at the same time 
    single_df = load_data(args.input_dir + "/dms_single.xlsx", type='single')
    double_df = load_data(args.input_dir + '/dms_pair.xlsx', type='pair')

    # remove entires that are too erroneous between single and pair measurements.
    _,_, error_inter_single, error_inter_pair = compare_single_fitness(single_df, double_df, args.error_threshold)
    double_df = double_df[~double_df.index.isin(error_inter_pair)]
    single_df = single_df[~single_df.index.isin(error_inter_single)]

    # parse the data
    single_df['Code'] = single_df['WT AA'] + single_df['Ambler Position'].astype(int).astype(str) + single_df['Mutant AA']
    processed_single_data = single_df[['Code', 
                                       'Fitness', 'Estimated error in fitness']]
    double_df['Code'] = double_df['WT AA 1'] + double_df['WT AA 2'] + double_df['Ambler Position'].astype(int).astype(str) + double_df['Mut AA 1'] + double_df['Mut AA 2']
    processed_data = double_df[['Code', 
                                'Mut 1 Fitness','Mut 1 Fitness Error', 
                                'Mut 2 Fitness', 'Mut 2 Fitness Error', 
                                'Double Mutant Fitness', 'Double Mutant Fitness Error',
                                'Epistasis']]

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    processed_data.to_csv(args.output_file, index=False)
    print(f"Processed data saved to {args.output_file}")