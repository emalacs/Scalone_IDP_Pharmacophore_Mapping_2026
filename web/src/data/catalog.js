// Colors mirror output/*/view_pharmacophore.pml (color commands).
// isoValue (contour level) is not fixed here — it's user-controlled via the
// contour slider, defaulted per-volume from its own stats once loaded. The
// pml's own 0.3 threshold for full-resolution maps currently produces an
// empty isosurface for these subsets: growth_features/*.mrc data (Aug 11)
// tops out around 0.22-0.55 depending on subset/feature, below the 0.3 cutoff
// the regenerated pml (Sep 2) hardcodes for 5 of 6 feature types.
//
// Feature identity (which chemical interaction type) is space-agnostic; the
// underlying directory and filename differ per SPACE below.
export const FEATURES = [
  { id: 'aromatic', label: 'Aromatic', color: 0x000000 },
  { id: 'hydrophobic', label: 'Hydrophobic', color: 0x228b22 },
  { id: 'hbond_acceptor', label: 'H-bond acceptor', color: 0xb22222 },
  { id: 'hbond_donor', label: 'H-bond donor', color: 0xb22222 },
  { id: 'negative_charge', label: 'Negative charge', color: 0xff0000 },
  { id: 'positive_charge', label: 'Positive charge', color: 0x0000ff },
]

// Two independent pharmacophores per subset, per view_pharmacophore.pml's own
// section headers: "CHEMICAL FEATURES (GROWTH SPACE)" (growth_features/) and
// "PHARMACOPHORE FROM CONTESTED SPACE" (pharmacophore_contested/). Growth
// space = protein occupancy < 30%, contested space = occupancy >= 30% (see
// negative_space/ comments in the same pml). baseNames map each FEATURES id
// to that space's actual filename prefix.
//
// pharmacophore_contested/ also has _contacts / _non_contacts / _diff /
// _ratio_contact_dominant / _ratio_noncontact_dominant variants beyond what's
// wired up here — only the plain occupancy maps (matching growth_features'
// level of detail) are exposed for now.
//
// Thinned top-N% variants are missing for some subsets in each space (source
// pipeline gap, not a bug here): growth space -- 1AA_PCA_C0_Graph0/1,
// Maso_PCA_C1_Graph0-3; contested space -- 1AA_PCA_C0(+Graph0/1),
// EPI_PCA_C1(+Graph0-3), Maso_PCA_C1(+Graph0-3). Picking an unavailable
// resolution surfaces the viewer's normal fetch-error state.
export const SPACES = [
  {
    id: 'growth',
    label: 'Growth space',
    featuresDir: 'growth_features',
    baseNames: {
      aromatic: 'aromatic_sites',
      hydrophobic: 'hydrophobic_sites',
      hbond_acceptor: 'hbond_acceptor_sites',
      hbond_donor: 'hbond_donor_sites',
      negative_charge: 'negative_charge_sites',
      positive_charge: 'positive_charge_sites',
    },
  },
  {
    id: 'contested',
    label: 'Contested space',
    featuresDir: 'pharmacophore_contested',
    baseNames: {
      aromatic: 'aromatic_occupancy',
      hydrophobic: 'hydrophobic_occupancy',
      hbond_acceptor: 'hbond_acceptors_occupancy',
      hbond_donor: 'hbond_donors_occupancy',
      negative_charge: 'negative_occupancy',
      positive_charge: 'positive_occupancy',
    },
  },
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

// All 47 curated subsets from output/ (everything with growth_features/,
// pharmacophore_contested/, negative_space/, and ligand_centroid.pdb;
// excludes output/graph_figures/ and output/full/ -- the latter has ligand
// resname "UNL" and no .pml/figures tying it to a known compound, so it was
// left out rather than guessed at). See the SPACES comment above for
// per-subset resolution-availability gaps.
export const SUBSETS = [
  {
    id: '1AA_full',
    label: '1AA · Full trajectory',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C0',
    label: '1AA · PCA C0 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C0_Graph0',
    label: '1AA · PCA C0 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C0_Graph1',
    label: '1AA · PCA C0 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C1',
    label: '1AA · PCA C1 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C1_Graph0',
    label: '1AA · PCA C1 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C1_Graph1',
    label: '1AA · PCA C1 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C1_Graph2',
    label: '1AA · PCA C1 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C1_Graph3',
    label: '1AA · PCA C1 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C2',
    label: '1AA · PCA C2 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C2_Graph0',
    label: '1AA · PCA C2 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C2_Graph1',
    label: '1AA · PCA C2 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C3',
    label: '1AA · PCA C3 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C3_Graph0',
    label: '1AA · PCA C3 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: '1AA_PCA_C3_Graph1',
    label: '1AA · PCA C3 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_full',
    label: 'EPI · Full trajectory',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C0',
    label: 'EPI · PCA C0 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C0_Graph0',
    label: 'EPI · PCA C0 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C0_Graph1',
    label: 'EPI · PCA C0 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C0_Graph2',
    label: 'EPI · PCA C0 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C0_Graph3',
    label: 'EPI · PCA C0 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C1',
    label: 'EPI · PCA C1 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C1_Graph0',
    label: 'EPI · PCA C1 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C1_Graph1',
    label: 'EPI · PCA C1 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C1_Graph2',
    label: 'EPI · PCA C1 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C1_Graph3',
    label: 'EPI · PCA C1 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C2',
    label: 'EPI · PCA C2 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C2_Graph0',
    label: 'EPI · PCA C2 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C2_Graph1',
    label: 'EPI · PCA C2 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C2_Graph2',
    label: 'EPI · PCA C2 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'EPI_PCA_C2_Graph3',
    label: 'EPI · PCA C2 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Masofaniten_full',
    label: 'Masofaniten · Full trajectory',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C0',
    label: 'Masofaniten · PCA C0 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C0_Graph0',
    label: 'Masofaniten · PCA C0 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C0_Graph1',
    label: 'Masofaniten · PCA C0 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C0_Graph2',
    label: 'Masofaniten · PCA C0 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C0_Graph3',
    label: 'Masofaniten · PCA C0 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C1',
    label: 'Masofaniten · PCA C1 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C1_Graph0',
    label: 'Masofaniten · PCA C1 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C1_Graph1',
    label: 'Masofaniten · PCA C1 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C1_Graph2',
    label: 'Masofaniten · PCA C1 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C1_Graph3',
    label: 'Masofaniten · PCA C1 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C2',
    label: 'Masofaniten · PCA C2 · Full cluster',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C2_Graph0',
    label: 'Masofaniten · PCA C2 · Graph 0',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C2_Graph1',
    label: 'Masofaniten · PCA C2 · Graph 1',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C2_Graph2',
    label: 'Masofaniten · PCA C2 · Graph 2',
    ligandFile: 'ligand_centroid.pdb',
  },
  {
    id: 'Maso_PCA_C2_Graph3',
    label: 'Masofaniten · PCA C2 · Graph 3',
    ligandFile: 'ligand_centroid.pdb',
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

export function featureUrl(subset, space, feature, resolution) {
  const baseName = space.baseNames[feature.id]
  return `${DATA_BASE}/${subset.id}/${space.featuresDir}/${baseName}${resolution.suffix}.mrc.gz`
}

export function negativeSpaceUrl(subset, layer) {
  return `${DATA_BASE}/${subset.id}/negative_space/${layer.file}`
}
