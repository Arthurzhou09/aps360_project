import pandas as pd
import argparse
import os
import sys
sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_utils import load_data, compare_single_fitness


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing the .xlsx files containing the DMS data')
    parser.add_argument('--output_dir', type=str, default='/Users/arthurzhou/github/aps360_project/src/data/processed', help='Path to save the processed data as CSV files')
    parser.add_argument('--error_threshold', type=float, default=100.0, help='Error threshold for filtering double mutant entries based on single mutant fitness values. Large values will keep more entries')
    parser.add_argument('--process_pair', action='store_true', help='Flag to indicate whether to process pair data. If not set, only single mutant data will be processed.')
    args = parser.parse_args()

    # processing should load both dfs in at the same time 
    single_df = load_data(args.input_dir + "/dms_single.xlsx", type='single')
    double_df = load_data(args.input_dir + '/dms_pair.xlsx', type='pair')

    # normalize ambler positions
    single_df['Ambler Position'] = single_df['Ambler Position'].astype(int) - single_df['Ambler Position'].min()
    double_df['Ambler Position'] = double_df['Ambler Position'].astype(int) - double_df['Ambler Position'].min()

    # remove entires that are too erroneous between single and pair measurements.TODO: decide how to formulate this approach if we want
    #_, stats, error_inter_single, error_inter_pair = compare_single_fitness(single_df, double_df, args.error_threshold)
    #double_df = double_df[~double_df.index.isin(error_inter_pair)]
    #single_df = single_df[~single_df.index.isin(error_inter_single)]

    # parse the data
    single_df['Code'] = single_df['WT AA'] + "_" + single_df['Ambler Position'].astype(int).astype(str) + "_" + single_df['Mutant AA']
    single_df['Single'] = 1
    processed_single_data = single_df[['Single', 'Ambler Position', 'Code', 
                                       'Fitness', 'Estimated error in fitness']].copy()
    processed_single_data['Epistasis'] = -111
    processed_single_data['Experiment Sequence'] = "".join(single_df.drop_duplicates(subset=['Ambler Position', 'WT AA'], ignore_index=True)['WT AA'].to_numpy())


    if args.process_pair:
        double_df['Code'] = double_df['WT AA 1'] + "_" + double_df['WT AA 2'] + "_" + double_df['Ambler Position'].astype(int).astype(str) + "_" + double_df['Mut AA 1'] + "_" + double_df['Mut AA 2']
        double_df['Single'] = 0
        processed_pair_data = double_df[['Single', 'Ambler Position', 'Code', 
                                    'Double Mutant Fitness', 'Double Mutant Fitness Error',
                                    'Epistasis']].rename(columns={
                                        'Double Mutant Fitness': 'Fitness',
                                        'Double Mutant Fitness Error': 'Estimated error in fitness'}).reset_index(drop=True)
        
        processed_data = pd.concat([processed_single_data, processed_pair_data], ignore_index=True)
    else:
        processed_data = processed_single_data

    os.makedirs(os.path.join(args.output_dir, f"err_{args.error_threshold}"), exist_ok=True)
    processed_data.to_csv(os.path.join(args.output_dir, f"err_{args.error_threshold}", f"dms_processed.csv"), index=False)
    print(f"Processed data saved to {args.output_dir}")