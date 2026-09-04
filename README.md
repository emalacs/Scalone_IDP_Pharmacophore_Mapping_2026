# Dynamic Pharmacophore Mapping for Intrinsically Disordered Drug Targets

Code and supplementary data accompanying the paper *"Dynamic Pharmacophore Mapping for Intrinsically Disordered Drug Targets."*

This pipeline defines pharmacophores from molecular dynamics simulations of intrinsically disordered proteins (IDPs): it extracts chemical interaction patterns (aromatic, hydrophobic, H-bond donor/acceptor, charged) between a small-molecule ligand and the IDP ensemble, computes per-voxel contact/occupancy probabilities across the trajectory, and produces volumetric pharmacophore maps.

**Live viewer:** https://emalacs.github.io/Scalone_IDP_Pharmacophore_Mapping_2026/

An interactive, browser-based viewer (React + [Mol*](https://molstar.org/)) for exploring the resulting pharmacophore density maps directly against the ligand structure — no local software required.

## Repository layout

```
web/      Interactive viewer (React + Vite + Mol*), deployed to GitHub Pages
output/   Pharmacophore density maps (.mrc, gzip-compressed) and ligand structures (.pdb)
          per trajectory subset, browsable and downloadable directly from this repo
```

`output/<subset>/` contains:
- `ligand_centroid.pdb` — representative ligand structure for that subset
- `growth_features/*.mrc.gz` — per-feature-type contact-probability maps (aromatic, hydrophobic, H-bond donor/acceptor, positive/negative charge), each at full resolution and four thinned top-N% variants
- `negative_space/*.mrc.gz` — solvent growth-space / contested-space occupancy maps

## Status

This repository currently contains the interactive viewer and a curated subset of pharmacophore maps (3 trajectory subsets). Still to come:

- Full analysis pipeline and notebooks (`pharmacophore_utils.py` and associated Jupyter notebooks), with local filesystem paths removed
- Pharmacophore maps for the remaining trajectory subsets
- A Google Colab notebook for running the pipeline on your own trajectories
- Raw MD trajectories, to be deposited on Zenodo (linked here once available)

## Running the viewer locally

```
cd web
npm install
npm run dev
```

## License

MIT — see [LICENSE](LICENSE).
