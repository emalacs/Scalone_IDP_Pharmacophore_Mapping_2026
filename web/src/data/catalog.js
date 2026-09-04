// Colors mirror output/*/view_pharmacophore.pml (color commands).
// isoValue (contour level) is not fixed here — it's user-controlled via the
// contour slider, defaulted per-volume from its own stats once loaded. The
// pml's own 0.3 threshold for full-resolution maps currently produces an
// empty isosurface for these subsets: growth_features/*.mrc data (Aug 11)
// tops out around 0.22-0.55 depending on subset/feature, below the 0.3 cutoff
// the regenerated pml (Sep 2) hardcodes for 5 of 6 feature types.
export const FEATURES = [
  { id: 'aromatic_sites', label: 'Aromatic sites', baseName: 'aromatic_sites', color: 0x000000 },
  { id: 'hydrophobic_sites', label: 'Hydrophobic sites', baseName: 'hydrophobic_sites', color: 0x228b22 },
  { id: 'hbond_acceptor_sites', label: 'H-bond acceptor sites', baseName: 'hbond_acceptor_sites', color: 0xb22222 },
  { id: 'hbond_donor_sites', label: 'H-bond donor sites', baseName: 'hbond_donor_sites', color: 0xb22222 },
  { id: 'negative_charge_sites', label: 'Negative charge sites', baseName: 'negative_charge_sites', color: 0xff0000 },
  { id: 'positive_charge_sites', label: 'Positive charge sites', baseName: 'positive_charge_sites', color: 0x0000ff },
]

// Matches the file variants view_pharmacophore.pml loads per feature: the
// full-resolution map (pml isomesh threshold 0.3) and four pre-thinned
// top-N% maps (pml isomesh threshold 0.01).
export const RESOLUTIONS = [
  { id: 'full', label: 'Full resolution', suffix: '' },
  { id: 'top20pct', label: 'Top 20%', suffix: '_top20pct' },
  { id: 'top10pct', label: 'Top 10%', suffix: '_top10pct' },
  { id: 'top5pct', label: 'Top 5%', suffix: '_top5pct' },
  { id: 'top1pct', label: 'Top 1%', suffix: '_top1pct' },
]

// Solvent-occupancy layers, shown alongside (not instead of) the selected
// feature map. In view_pharmacophore.pml these are PyMOL `volume` objects with
// a custom color ramp rather than `isomesh`; here they render as a second and
// third isosurface layered into the same scene. Colors approximate the pml
// ramp endpoints (growth_vol -> blue, contested_vol -> orange).
export const NEGATIVE_SPACE_LAYERS = [
  { id: 'growth_space', label: 'Growth space', file: 'growth_space.mrc.gz', color: 0x2255ee },
  { id: 'contested_space', label: 'Contested space', file: 'contested_space.mrc.gz', color: 0xff7800 },
]

export const SUBSETS = [
  {
    id: 'EPI_PCA_C2_Graph0',
    label: 'EPI · PCA C2 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
    featuresDir: 'growth_features',
  },
  {
    id: 'EPI_PCA_C2_Graph1',
    label: 'EPI · PCA C2 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
    featuresDir: 'growth_features',
  },
  {
    id: 'EPI_PCA_C2_Graph2',
    label: 'EPI · PCA C2 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
    featuresDir: 'growth_features',
  },
]

// .mrc/.pdb data lives in this repo's own output/ folder (not bundled into
// the web/ app) so it stays independently browsable and downloadable on
// GitHub, and so the Pages build stays small regardless of how much data is
// added later. Override via VITE_DATA_BASE_URL for local testing against a
// different branch/fork.
const DATA_BASE =
  import.meta.env.VITE_DATA_BASE_URL ||
  'https://raw.githubusercontent.com/emalacs/Scalone_IDP_Pharmacophore_Mapping_2026/main/output'

export function ligandUrl(subset) {
  return `${DATA_BASE}/${subset.id}/${subset.ligandFile}`
}

export function featureUrl(subset, feature, resolution) {
  return `${DATA_BASE}/${subset.id}/${subset.featuresDir}/${feature.baseName}${resolution.suffix}.mrc.gz`
}

export function negativeSpaceUrl(subset, layer) {
  return `${DATA_BASE}/${subset.id}/negative_space/${layer.file}`
}
