import sys
sys.path.append('/Users/arthurzhou/github/aps360_project')
from data.data_utils import load_aa_index
import argparse
from glob import glob
import os
import pandas as pd

if __name__ == "__main__":
    parser =argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=False, help='Path to the directory containing the .ttl files with record ids')
    parser.add_argument('--output_dir', type=str, default='/Users/arthurzhou/github/aps360_project/src/data/process/', help='Path to save the processed data as a CSV file')
    args = parser.parse_args()

    aa_values ={}
    all_files = glob(args.input_dir + "/*.ttl")
    for file in all_files:
        Id = file.split("/")[-1].split(".")[0]
        data, values = load_aa_index(Id)
        aa_values[Id] = (values, data['description'])

    # save as df to be same as dms data...
    result= pd.DataFrame.from_dict(
        {idx: { **values,
                "description": description
            }for idx, (values, description) in aa_values.items()
        },
        orient="index"
    ).reset_index(names="id")

    os.makedirs(args.output_dir, exist_ok=True) 
    result.to_csv(os.path.join(args.output_dir, "aa_index_data.csv"), index=False)
    print(f"Processed data saved to {os.path.join(args.output_dir, 'aa_index_data.csv')}")

    