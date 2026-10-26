# Paper 3 — environmental structure of *Austropotamobius torrentium* lineages

Multi-class comparison of four LABEL groups: *A. torrentium* CSE, SB and NCD (NCD merged
from seven haplogroups) and *A. bihariensis* as reference, on the GeoTraits variables.
Input is `data/raw/master_lineage.xlsx` (not tracked).

Self-contained copy of the pipeline, like `../conservation/`. Run from inside this
folder; `data` links to the repository's shared `data/` folder:

```bash
cd lineage
../.venv/bin/python src/random_forest_lineage.py --input data/raw/master_lineage.xlsx
```

Outputs go to `results_lineage/`. Scripts specific to this paper: `decision_tree_lineage.py`,
`random_forest_lineage.py`, `niche_overlap_lineage.py`, `ncd_haplogroup_pca.py`,
`lineage_pairwise_summary.py`, `robustness_checks_lineage.py`,
`supplementary_tables_lineage.py`, `export_lineage_binary_datasets.py`. The other
modules are the inherited pipeline.

Merged into `main` from the former branch `lineage-paper` (43ba41e, 13 April 2026),
which is also what github.com/KristianMiok/niche-lineage holds.
