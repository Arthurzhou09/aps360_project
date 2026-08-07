### KYTJ820101 (hydropathy)
The Kyte-Doolittle scale is widely used for detecting hydrophobic regions in proteins. Regions with a positive value are hydrophobic. This scale can be used for identifying both surface-exposed regions as well as transmembrane regions, depending on the window size used. Short window sizes of 5-7 generally work well for predicting putative surface-exposed regions. Large window sizes of 19-21 are well suited for finding transmembrane domains if the values calculated are above 1.6 [Kyte and Doolittle, 1982]. These values should be used as a rule of thumb and deviations from the rule may occur.

### CHAM810101 (sterics)
Steric parameter (Charton, 1981). Higher scores mean more steric bullkiness.

### CHOC760101  (solvent accesibility) OR JANJ780101
Theoretical intrinsic exposure capacity in a standardized peptide (tripepetide normalized). Performs Gly–X–Gly and measures SA of X residue in its unfolded local state. Higher scores mean more exposed SA. This is an idealized model and does not account other influences on exposure.
Alternativly, JANJ780101 uses average observed in proteins which includes folding constraints. It is more realistic, but hard to decouple other effects that may influence scores.

### VINM940101 (flexibility, B-scores)
x-ray crystallography derived intrinsic flexibility statiscally normalized (b-score). Higher scores mean more local flexibility/atom displacement.

### GRAR740102 (polarity)
Amino acid difference formula to help explain protein evolution. May affect solvent interactions

### KLEP840101 (net charge)
Affects hydrolysis. 

### CHOP780202 ($/beta sheet propensity)
Normalized frequency of beta-sheet. Tem-beta has aalpha beta scaffold.

### CHOP780201 (alpha-helix propensity)
Normalized frequency of alpha-helix. Tem-beta has alpha beta scaffold.

### *NEW* Use PCA condensed AAindex
Taken from: https://github.com/gitter-lab/nn4dms/tree/master/data/aaindex



### PF00144 (Beta-lactamase family, Pfam/InterPro)
Used for self-supervised structure-conditioned pretraining: homolog sequences are threaded onto the TEM-1 (1BTL) structure graph and the GNN is trained to recover a masked residue's identity from its neighbours, with no fitness labels involved. TEM-1 belongs to this family (InterPro IPR001466). Family page: https://www.ebi.ac.uk/interpro/entry/pfam/PF00144/

Downloads (via `homolog_process.py`):
- Full alignment (61,457 sequences, Stockholm, gzipped) — save as `PF00144_full.sto.gz`:
  https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00144/?annotation=alignment:full
- Seed alignment (127 curated sequences, much smaller, useful for a quick test run) — save as `PF00144_seed.sto.gz`:
  https://www.ebi.ac.uk/interpro/api/entry/pfam/PF00144/?annotation=alignment:seed

Fallback: UniRef90, ~30GB compressed, https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.: prob not doing this this is too much.


### MSA 
go to: https://proteingym.org/download
After unzipping, find `BLAT_ECOLX_full_11-26-2021_b02.a2m`