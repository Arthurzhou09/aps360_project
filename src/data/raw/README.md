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
Norrmalized frequency of beta-sheet. Tem-beta has aalpha beta scaffold.

### CHOP780201 (alpha-helix propensity)
Normalized frequency of alpha-helix. Tem-beta has aalpha beta scaffold.