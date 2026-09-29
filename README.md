# Dynamic Pharmacophore Mapping for Intrinsically Disordered Drug Targets

Code and supplementary data accompanying the paper *"Dynamic Pharmacophore Mapping for Intrinsically Disordered Drug Targets."*

This pipeline defines pharmacophores from molecular dynamics simulations of intrinsically disordered proteins (IDPs): it extracts chemical interaction patterns (aromatic, hydrophobic, H-bond donor/acceptor, charged) between a small-molecule ligand and the IDP ensemble, computes per-voxel contact/occupancy probabilities across the trajectory, and produces volumetric pharmacophore maps.

**Live viewer:** https://emalacs.github.io/Scalone_IDP_Pharmacophore_Mapping_2026/

An interactive, browser-based viewer (React + [Mol*](https://molstar.org/)) for exploring the resulting pharmacophore density maps directly against the ligand structure — no local software required.

## Repository layout

```
web/       Interactive viewer (React + Vite + Mol*), deployed to GitHub Pages
output/    Pharmacophore density maps (.mrc, gzip-compressed) and ligand structures (.pdb)
           per trajectory subset, browsable and downloadable directly from this repo
notebooks/ Notebooks for running the pipeline on your own trajectories -- local by default,
           notebooks/colab/ has the Google Colab variants (upload widget, no local install)
pharmacophore_utils.py  Core pipeline module the notebooks import
```

`output/<subset>/` contains:
- `ligand_centroid.pdb` — representative ligand structure for that subset
- `growth_features/*.mrc.gz` — per-feature-type contact-probability maps (aromatic, hydrophobic, H-bond donor/acceptor, positive/negative charge), each at full resolution and four thinned top-N% variants
- `negative_space/*.mrc.gz` — solvent growth-space / contested-space occupancy maps

## Running the pipeline on your own trajectory

**Locally** (installs [`idp-pharmacophore-tools`](https://github.com/emalacs/idp-pharmacophore-tools), points at files already on disk):

[`01_split_trajectory_pca_graph.ipynb`](notebooks/01_split_trajectory_pca_graph.ipynb) — split a trajectory + topology into conformational subsets by PCA and graph clustering

[`02_pharmacophore_from_subset.ipynb`](notebooks/02_pharmacophore_from_subset.ipynb) — compute a single trajectory's (e.g. one subset from the notebook above) pharmacophore maps and a PyMOL script to view them

**Or in Google Colab** (no local install, upload widget instead of local paths):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emalacs/Scalone_IDP_Pharmacophore_Mapping_2026/blob/main/notebooks/colab/01_split_trajectory_pca_graph.ipynb) [`notebooks/colab/01_split_trajectory_pca_graph.ipynb`](notebooks/colab/01_split_trajectory_pca_graph.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/emalacs/Scalone_IDP_Pharmacophore_Mapping_2026/blob/main/notebooks/colab/02_pharmacophore_from_subset.ipynb) [`notebooks/colab/02_pharmacophore_from_subset.ipynb`](notebooks/colab/02_pharmacophore_from_subset.ipynb)

Each notebook's own intro cell links to its counterpart if you started in the wrong one.

## Status

This repository currently contains the interactive viewer, a curated subset of pharmacophore maps (3 trajectory subsets), and the analysis pipeline + notebooks (local and Colab). Still to come:

- Pharmacophore maps for the remaining trajectory subsets
- Raw MD trajectories, to be deposited on Zenodo (linked here once available)

## Running the viewer locally

```
cd web
npm install
npm run dev
```

## License

MIT — see [LICENSE](LICENSE).
