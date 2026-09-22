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
notebooks/ Google Colab notebooks for running the pipeline on your own trajectories
pharmacophore_utils.py  Core pipeline module the notebooks import
```

`output/<subset>/` contains:
- `ligand_centroid.pdb` — representative ligand structure for that subset
- `growth_features/*.mrc.gz` — per-feature-type contact-probability maps (aromatic, hydrophobic, H-bond donor/acceptor, positive/negative charge), each at full resolution and four thinned top-N% variants
- `negative_space/*.mrc.gz` — solvent growth-space / contested-space occupancy maps

## Running the pipeline on your own trajectory

Open in Google Colab (no local install required):
- [`notebooks/01_split_trajectory_pca_graph.ipynb`](notebooks/01_split_trajectory_pca_graph.ipynb) — upload a trajectory + topology, split it into conformational subsets by PCA and graph clustering
- [`notebooks/02_pharmacophore_from_subset.ipynb`](notebooks/02_pharmacophore_from_subset.ipynb) — upload a single trajectory (e.g. one subset from the notebook above), compute its pharmacophore maps and a PyMOL script to view them

Or run `pharmacophore_utils.py` locally — see `requirements.txt`.

## Status

This repository currently contains the interactive viewer, a curated subset of pharmacophore maps (3 trajectory subsets), and the analysis pipeline + Colab notebooks. Still to come:

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
