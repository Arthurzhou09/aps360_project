import pandas as pd
import argparse
import os
import sys
sys.path.append('/Users/arthurzhou/github/aps360_project')
from data.data_utils import load_data, compare_single_fitness



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing the .xlsx files containing the DMS data')
    parser.add_argument('--output_dir', type=str, default='/Users/arthurzhou/github/aps360_project/src/data/process', help='Path to save the processed data as CSV files')
    parser.add_argument('--error_threshold', type=float, default=0.0, help='Error threshold for filtering double mutant entries based on single mutant fitness values')
    args = parser.parse_args()

    # processing should load both dfs in at the same time 
    single_df = load_data(args.input_dir + "/dms_single.xlsx", type='single')
    double_df = load_data(args.input_dir + '/dms_pair.xlsx', type='pair')

    # remove entires that are too erroneous between single and pair measurements.
    _,_, error_inter_single, error_inter_pair = compare_single_fitness(single_df, double_df, args.error_threshold)
    double_df = double_df[~double_df.index.isin(error_inter_pair)]
    single_df = single_df[~single_df.index.isin(error_inter_single)]

    # parse the data
    single_df['Code'] = single_df['WT AA'] + "_" + single_df['Ambler Position'].astype(int).astype(str) + "_" + single_df['Mutant AA']
    single_df['Single'] = 1
    processed_single_data = single_df[['Single', 'Code', 
                                       'Fitness', 'Estimated error in fitness']]
    processed_single_data['Epistatsis'] = -111


    double_df['Code'] = double_df['WT AA 1'] + "_" + double_df['WT AA 2'] + "_" + double_df['Ambler Position'].astype(int).astype(str) + "_" + double_df['Mut AA 1'] + "_" + double_df['Mut AA 2']
    double_df['Single'] = 0
    processed_pair_data = double_df[['Single', 'Code', 
                                'Double Mutant Fitness', 'Double Mutant Fitness Error',
                                'Epistasis']].rename(columns={
                                    'Double Mutant Fitness': 'Fitness',
                                    'Double Mutant Fitness Error': 'Estimated error in fitness'})
    
    processed_data = pd.concat([processed_single_data, processed_pair_data], ignore_index=True)

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)
    processed_data.to_csv(os.path.join(args.output_dir, "dms_processed.csv"), index=False)
    print(f"Processed data saved to {args.output_file}")