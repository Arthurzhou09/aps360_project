# src/data

## Modules

- `data_class.py` — Base `Dataset`/`Data` classes (`DataClass`, `ProteinGraphData`, `HomologGraphData`) shared by all dataset implementations.
- `data_utils.py` — Shared I/O and preprocessing helpers: loading DMS/AAindex/structure/alignment files, sequence alignment, mutation masking.
- `feature_utils.py` — Feature engineering: k-NN edge construction, RBF distance embeddings, node feature assembly, AAindex encoding, DCI perturbation matrix/edges.
- `dfi_calc.py` — Vendored Dynamic Flexibility/Coupling Index (DFI/DCI) implementation (elastic network model), used by `feature_utils.py`.
- `tem_beta.py` — `Tem1BetaLactamaseDataset` / `MLPDataset`: supervised datasets feeding DMS mutant graphs (or flattened features) and fitness labels to the GNN/MLP.
- `homolog.py` — `HomologMaskedDataset`: masked-residue SSL dataset built from Pfam homolog sequences threaded onto the WT structure graph.
- `msa.py` — `MSAMaskedDataset`: masked-residue SSL dataset built from an MSA's own match columns (corrected successor to `homolog.py`).
- `split.py` — Train/val/test split strategies: by structural position, held-out double mutants, or homolog/cluster.

## `process/` (raw -> processed CSVs, run as standalone scripts)

- `dms_process.py` — Cleans raw DMS Excel sheets (single/pair mutants) into `dms_processed.csv`.
- `aa_index_process.py` — Parses AAindex `.ttl` records into a single amino-acid property CSV.
- `homolog_process.py` — Filters a Pfam Stockholm alignment to WT homologs by identity (superseded by `msa_process.py`).
- `msa_process.py` — Converts a ProteinGym-style a2m alignment into per-column match-state data mapped onto the WT structure.
- `pssm_process.py` — Builds a per-position PSSM (amino acid frequencies) from the homolog set, used by the evolutionary baseline.
