import argparse
import os
import sys
sys.path.append(r"C:\Users\Arthur Zhou\GitHub\aps360_project\src")
from data.data_utils import load_alignment, extract_family_sequences, filter_homologs_by_identity, load_cif_structure, parse_structure


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='./src/data/raw', help='Path to the directory containing the Pfam alignment file (.sto or .sto.gz)')
    parser.add_argument('--alignment_file', type=str, default='PF00144.alignment.full', help='Name of the Pfam full alignment file within input_dir')
    parser.add_argument('--pdb_dir', type=str, default='C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed', help='Path to the directory containing the WT PDB/CIF structure')
    parser.add_argument('--pdb_id', type=str, default='1BTL', help='PDB ID of the WT reference structure')
    parser.add_argument('--min_identity', type=float, default=0.4, help='Minimum fraction identity to WT to keep a homolog sequence')
    parser.add_argument('--max_identity', type=float, default=0.99, help='Maximum fraction identity to WT to keep a homolog sequence. Excludes near-exact duplicates of WT itself')
    parser.add_argument('--max_sequences', type=int, default=None, help='Optional cap on number of alignment records to process, useful for a quick test run before processing the full family')
    parser.add_argument('--output_dir', type=str, default='C:\\Users\\Arthur Zhou\\GitHub\\aps360_project\\src\\data\\processed\\homologs', help='Path to save the processed data as a CSV file')
    args = parser.parse_args()

    wt_sequence, _ = parse_structure(load_cif_structure(f"{args.pdb_dir}/{args.pdb_id}.cif", args.pdb_id))

    alignment = load_alignment(os.path.join(args.input_dir, args.alignment_file))
    if args.max_sequences is not None:
        alignment = alignment[:args.max_sequences]
    print(f"Loaded {len(alignment)} records from {args.alignment_file}")

    sequences = extract_family_sequences(alignment)
    print(f"{len(sequences)} sequences kept after dropping non-canonical residues")

    homolog_df = filter_homologs_by_identity(sequences, wt_sequence, args.min_identity, args.max_identity)
    print(f"{len(homolog_df)} homologs kept within [{args.min_identity}, {args.max_identity}] identity to {args.pdb_id}")

    os.makedirs(args.output_dir, exist_ok=True)
    homolog_df.to_csv(os.path.join(args.output_dir, "homolog_processed.csv"), index=False)
    print(f"Processed data saved to {os.path.join(args.output_dir, 'homolog_processed.csv')}")
