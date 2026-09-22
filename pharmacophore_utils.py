"""
pharmacophore_utils.py

Pharmacophore definition from MD simulations of intrinsically disordered proteins.

Hybrid library strategy:
  - MDAnalysis + RDKit : topology inspection, ligand extraction, chemical feature typing
  - MDTraj             : trajectory loading and all distance/contact computation

Usage as a module:
    import pharmacophore_utils as pu
    sim = pu.PharmacophoreTrajectory("system.gro", "traj.xtc")
    sim.load(stride=1)

Usage from the CLI:
    python pharmacophore_utils.py --topology system.gro --trajectory traj.xtc
"""

import argparse
import itertools
import math
import os
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from itertools import product
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available — install: pip install pandas")

pyblock = None
try:
    import pyblock
    PYBLOCK_AVAILABLE = True
except ImportError:
    PYBLOCK_AVAILABLE = False
    print("Warning: pyblock not available — block errors default to 0; "
          "install: pip install pyblock")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    tqdm = lambda x, **kw: x   # no-op passthrough when tqdm not installed

try:
    import mdtraj as md
    MDTRAJ_AVAILABLE = True
except ImportError:
    MDTRAJ_AVAILABLE = False
    print("Warning: MDTraj not available")

# MDTraj H-bond geometry helpers — private API used by baker_hubbard2.
# Standard _get_bond_triplets does NOT accept lig_donors; ligand donors are
# appended manually inside baker_hubbard2 after calling the standard function.
_get_bond_triplets        = None
_compute_bounded_geometry = None
try:
    from mdtraj.geometry.hbond import _get_bond_triplets, _compute_bounded_geometry
except ImportError:
    if MDTRAJ_AVAILABLE:
        print("Warning: mdtraj.geometry.hbond internals not importable — "
              "compute_hbond_contacts() will not work")

try:
    import MDAnalysis as mda
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False
    print("Warning: MDAnalysis not available")

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D
    import io
    from PIL import Image
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    print("Warning: RDKit/PIL not available")

# rdDetermineBonds was added in RDKit 2022.09 — keep separate so an older RDKit
# doesn't break RDKIT_AVAILABLE for the rest of the module.
rdDetermineBonds = None
try:
    from rdkit.Chem import rdDetermineBonds
    RDKIT_DETERMINE_BONDS_AVAILABLE = True
except ImportError:
    RDKIT_DETERMINE_BONDS_AVAILABLE = False
    if RDKIT_AVAILABLE:
        print("Warning: rdDetermineBonds not available (requires RDKit >= 2022.09) — "
              "aromaticity will not be detected correctly for .gro inputs")

go = None
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import torch
    USE_GPU    = torch.cuda.is_available()
    GPU_DEVICE = torch.device('cuda:1') if USE_GPU else None
    # GPU_DEVICE = torch.device('cuda') if USE_GPU else None
    TORCH_AVAILABLE = True
    if USE_GPU:
        print(f"[GPU] Acceleration enabled: {torch.cuda.get_device_name(0)}")
except ImportError:
    torch      = None
    USE_GPU    = False
    GPU_DEVICE = None
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available — GPU acceleration disabled for voxel analysis; "
          "install: pip install torch")

try:
    import mrcfile
    MRCFILE_AVAILABLE = True
except ImportError:
    mrcfile = None
    MRCFILE_AVAILABLE = False
    print("Warning: mrcfile not available — MRC volume export disabled; "
          "install: pip install mrcfile")

try:
    from sklearn.metrics import pairwise_distances as _sklearn_pairwise_distances
    from sklearn.metrics import silhouette_score as _sklearn_silhouette_score
    from sklearn.metrics import silhouette_samples as _sklearn_silhouette_samples
    from sklearn.decomposition import PCA as _sklearn_PCA
    from sklearn.preprocessing import StandardScaler as _sklearn_StandardScaler
    from sklearn.cluster import AgglomerativeClustering as _sklearn_AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    _sklearn_pairwise_distances       = None
    _sklearn_silhouette_score         = None
    _sklearn_silhouette_samples       = None
    _sklearn_PCA                      = None
    _sklearn_StandardScaler           = None
    _sklearn_AgglomerativeClustering  = None
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available — ligand centroid PDB, PCA, and graph "
          "clustering disabled; install: pip install scikit-learn")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    nx = None
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available — graph clustering disabled; "
          "install: pip install networkx")

try:
    import hdbscan as _hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    _hdbscan = None
    HDBSCAN_AVAILABLE = False
    # Only warn when actually requested — hdbscan is an optional clustering backend
    # for compute_graph_clustering()/cluster_graph_clustering() (default backend is
    # 'agglomerative', which only needs scikit-learn).

try:
    from scipy import ndimage as _scipy_ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    _scipy_ndimage = None
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available — Gaussian smoothing of MRC maps disabled; "
          "install: pip install scipy")

try:
    from deeptime.clustering import KMeans as _DeeptimeKMeans
    DEEPTIME_AVAILABLE = True
except ImportError:
    _DeeptimeKMeans = None
    DEEPTIME_AVAILABLE = False
    print("Warning: deeptime not available — PCA K-means clustering disabled; "
          "install: pip install deeptime")

try:
    import matplotlib.pyplot as _plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    _plt = None
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available — FES plots disabled; "
          "install: pip install matplotlib")

# ── Default settings ─────────────────────────────────────────────────────────
DEFAULT_LIGAND_RESNAME    = "LIG"
DEFAULT_AROMATIC_CUTOFF   = 5.5   # Å  (MDTraj uses nm: multiply by 0.1)
DEFAULT_CONTACT_CUTOFF      = 0.60   # nm (6 Å) — distance-based contact cutoff
DEFAULT_HYDROPHOBIC_CUTOFF  = 0.40   # nm (4.0 Å) — hydrophobic contact cutoff
DEFAULT_CONTACT_THRESHOLD = 0.30
DEFAULT_OUTPUT_DIR        = "./output"
DEFAULT_THREAD_WORKERS    = min(8, os.cpu_count() or 1)  # CPU-threaded voxel loops

# ── Residue classification (used by analyze_topology) ────────────────────────
_STANDARD_AA = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY',
    'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL', 'CYX', 'HID', 'HIE', 'HIP',
}
_CAPS   = {'ACE', 'NME', 'NHE', 'NMF', 'NH2', 'CT3', 'CT2', 'CT1'}
_WATER  = {'HOH', 'WAT', 'TIP3', 'SOL', 'TIP3P', 'TIP4P', 'SPC'}
_IONS   = {'NA', 'CL', 'K', 'CA', 'MG', 'ZN', 'NA+', 'CL-', 'K+', 'CA2+', 'MG2+'}
_NON_LIGAND = _STANDARD_AA | _CAPS | _WATER | _IONS


# ── Private helpers ───────────────────────────────────────────────────────────

def _threaded_frame_reduce(n_frames, partial_fn, combine_fn, n_workers=None):
    """Run partial_fn over disjoint, contiguous frame ranges in parallel threads
    and combine the per-range results.

    Used by the CPU "_threaded" sibling methods (compute_negative_space_threaded,
    compute_growth_space_features_threaded, compute_pharmacophore_maps_threaded)
    as a drop-in replacement for their serial "for fi in range(n_frames)" loop.
    Safe because each partial_fn(start, stop) call owns its own private
    accumulator array(s) — no shared mutable state is written concurrently —
    and because the heavy per-frame work (np.linalg.norm over large arrays)
    releases the GIL, so threads (not processes) give real parallelism here
    without the cost of pickling voxel/trajectory data across process boundaries.

    Parameters
    ----------
    n_frames : int
    partial_fn : Callable[[int, int], R]
        Called once per worker as partial_fn(start, stop) for a disjoint range
        of the original frame loop; returns a partial result (an ndarray, or a
        tuple of ndarrays, matching what combine_fn expects).
    combine_fn : Callable[[List[R]], R]
        Combines the list of per-worker partial results into the final result
        (e.g. elementwise sum for accumulators, elementwise min for running-min).
    n_workers : int, optional
        Defaults to DEFAULT_THREAD_WORKERS, capped at n_frames.

    Returns
    -------
    R : whatever combine_fn returns.
    """
    n_workers = max(1, min(n_workers or DEFAULT_THREAD_WORKERS, n_frames))
    bounds = np.linspace(0, n_frames, n_workers + 1).astype(int)
    ranges = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        results = list(executor.map(lambda ab: partial_fn(*ab), ranges))
    return combine_fn(results)


def _sum_reduce(parts):
    """combine_fn for _threaded_frame_reduce: elementwise sum of accumulator arrays."""
    return reduce(np.add, parts)


def _tuple_sum_reduce(parts):
    """combine_fn for _threaded_frame_reduce: elementwise sum, one array tuple per worker."""
    return tuple(reduce(np.add, arrs) for arrs in zip(*parts))


def _fix_element_cap(element: str) -> str:
    """Normalize element symbol capitalization (e.g. 'CL' → 'Cl', 'c' → 'C')."""
    return element.strip().capitalize()


# ════════════════════════════════════════════════════════════════════════════
# Standalone topology utilities (callable from notebook or load())
# ════════════════════════════════════════════════════════════════════════════

def analyze_topology(topology_path: str, trajectory_path: Optional[str] = None) -> Dict:
    """Scan a topology file to detect ligands, protein composition, and cysteines.

    Parameters
    ----------
    topology_path : str
        Path to .gro or .pdb topology file.
    trajectory_path : str, optional
        Trajectory path (required for some topology formats).

    Returns
    -------
    dict
        Keys: has_ligand, ligand_resname, ligand_atom_names, all_ligands,
        cysteines, n_atoms, n_residues. On failure: {'error': str}.
    """
    if not MDA_AVAILABLE:
        return {'error': 'MDAnalysis not available — install: pip install MDAnalysis'}
    try:
        args = [topology_path] + ([trajectory_path] if trajectory_path else [])
        u = mda.Universe(*args)
        ligands, cysteines = [], []
        for residue in u.residues:
            resname = residue.resname.strip().upper()
            if resname in ('CYS', 'CYX'):
                cysteines.append({'resid': int(residue.resid), 'resname': resname})
            if resname not in _NON_LIGAND:
                ligands.append({
                    'resname': resname,
                    'resid': int(residue.resid),
                    'n_atoms': len(residue.atoms),
                    'atom_names': [a.name for a in residue.atoms],
                })
        has_ligand = bool(ligands)
        first = ligands[0] if has_ligand else None
        return {
            'has_ligand': has_ligand,
            'ligand_resname': first['resname'] if first else None,
            'ligand_atom_names': first['atom_names'] if first else [],
            'all_ligands': ligands,
            'cysteines': cysteines,
            'n_atoms': int(len(u.atoms)),
            'n_residues': int(len(u.residues)),
        }
    except Exception as e:
        return {'error': str(e)}


def extract_ligand_mol(topology_path: str, ligand_resname: str,
                       trajectory_path: Optional[str] = None):
    """Extract the ligand from a topology file and return an RDKit Mol object.

    Uses MDAnalysis to read the topology (handles .gro element guessing),
    writes a temporary PDB with correct element columns, then reads with RDKit.

    Parameters
    ----------
    topology_path : str
        Path to .gro or .pdb topology file.
    ligand_resname : str
        Residue name of the ligand.
    trajectory_path : str, optional
        Trajectory path (used if topology lacks coordinates).

    Returns
    -------
    rdkit.Chem.Mol
    """
    if not MDA_AVAILABLE:
        raise ImportError("MDAnalysis required — install: pip install MDAnalysis")
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit required — install: conda install -c conda-forge rdkit")

    args = [topology_path] + ([trajectory_path] if trajectory_path else [])
    u = mda.Universe(*args)
    ligand = u.select_atoms(f"resname {ligand_resname}")

    if len(ligand) == 0:
        raise ValueError(f"No atoms found for ligand resname '{ligand_resname}'")

    print(f"Extracting ligand '{ligand_resname}': {len(ligand)} atoms")

    # Guess element types — .gro files lack element info
    try:
        elements = [_fix_element_cap(a.element) for a in ligand.atoms]
    except Exception:
        from MDAnalysis.topology.guessers import guess_types
        elements = [_fix_element_cap(e) for e in guess_types(ligand.names)]
        for atom, elem in zip(ligand.atoms, elements):
            atom.type = elem

    # Write temporary PDB with explicit element columns (cols 77-78)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pdb', delete=False) as tmp:
        tmp_path = tmp.name

    coords = ligand.positions
    with open(tmp_path, 'w') as f:
        for i, (atom, elem) in enumerate(zip(ligand.atoms, elements)):
            x, y, z = coords[i]
            f.write(
                f"ATOM  {i+1:5d}  {atom.name:<4s}LIG     1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}\n"
            )
        f.write("END\n")

    mol = Chem.MolFromPDBFile(tmp_path, removeHs=False, sanitize=False, proximityBonding=True)
    os.unlink(tmp_path)

    if mol is None:
        raise RuntimeError("RDKit could not parse the extracted ligand PDB")

    # .gro files carry no bond order info, so proximityBonding assigns all bonds as single.
    # DetermineBondOrders infers correct bond orders from 3D geometry + valence rules,
    # which is required for aromaticity perception to work during sanitization.
    # Assumes a neutral ligand (charge=0) — adjust if the ligand is charged.
    if rdDetermineBonds is not None:
        try:
            rdDetermineBonds.DetermineBondOrders(mol, charge=0)
        except Exception as e:
            print(f"Warning: bond order determination failed ({e}) — "
                  "aromaticity may not be detected correctly")

    try:
        Chem.SanitizeMol(mol, catchErrors=True)
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_KEKULIZE)
        except Exception:
            print("Warning: molecule could not be fully sanitized — some features may be limited")

    print(f"Ligand RDKit mol: {mol.GetNumAtoms()} atoms")
    return mol


def draw_molecule_with_labels(mol, width: int = 900, height: int = 900):
    """Draw the ligand as a 2D structure with atom indices labeled.

    Designed for inline display in Jupyter (returns a PIL Image — no file saved).

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        RDKit molecule (e.g. sim.ligand_mol or from extract_ligand_mol()).
    width, height : int
        Image dimensions in pixels.

    Returns
    -------
    (PIL.Image, rdkit.Chem.Mol) or None
    """
    if not RDKIT_AVAILABLE:
        print("Warning: RDKit/PIL not available")
        return None
    if mol is None:
        print("Warning: mol is None")
        return None
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.Compute2DCoords(mol)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                continue
            info = atom.GetMonomerInfo()
            name = info.GetName().strip() if info else ''
            if name:
                atom.SetProp('atomNote', name)
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.drawOptions().addAtomIndices = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        return img, mol
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return None


# ── Aromatic geometry helpers ─────────────────────────────────────────────────

def find_plane_normal_new(positions):
    """Fit a best-fit plane to each set of ring atoms and return normal vectors.

    Parameters
    ----------
    positions : np.ndarray, shape (n_frames, n_ring_atoms, 3)

    Returns
    -------
    np.ndarray, shape (n_frames, 3)
    """
    num_atoms = positions.shape[0]
    A = positions[:, :, 0:2].reshape(-1, 2)
    B = positions[:, :, 2].reshape(-1)
    A = np.concatenate((A, np.ones((A.shape[0], 1))), axis=1)
    A = A.reshape(num_atoms, positions.shape[1], -1)
    results = np.empty((num_atoms, 3))
    for i in range(num_atoms):
        A_slice = A[i, :, :]
        B_slice = B[i * A.shape[1]:(i + 1) * A.shape[1]]
        out = np.linalg.lstsq(A_slice, B_slice, rcond=-1)
        na_c, nb_c, d_c = out[0]
        if d_c != 0.0:
            cu = 1.0 / d_c
            bu = -nb_c * cu
            au = -na_c * cu
        else:
            cu, bu, au = 1.0, -nb_c, -na_c
        normal = np.asarray([au, bu, cu])
        normal /= np.linalg.norm(normal)
        results[i, :] = normal
    return results


def find_plane_normal2_assign_atomid_new(positions, id1, id2, id3):
    """Compute ring normal via cross product of three atoms (sign reference)."""
    v1 = positions[:, id1] - positions[:, id2]
    v1 /= np.linalg.norm(v1, axis=1)[:, np.newaxis]
    v2 = positions[:, id3] - positions[:, id1]
    v2 /= np.linalg.norm(v2, axis=1)[:, np.newaxis]
    return np.cross(v1, v2)


def get_ring_center_normal_trj_assign_atomid_new(position_array, id1, id2, id3):
    """Compute ring center and sign-consistent normal vector for every frame.

    Parameters
    ----------
    position_array : np.ndarray, shape (n_frames, n_ring_atoms, 3)
    id1, id2, id3 : int
        Atom positions within the ring used as the sign-reference normal.

    Returns
    -------
    np.ndarray, shape (n_frames, 2, 3)
        [:, 0, :] = ring center, [:, 1, :] = normal vector.
    """
    centers_overtraj = np.mean(position_array, axis=1)  # (n_frames, 3)
    normal  = find_plane_normal_new(position_array)
    normal2 = find_plane_normal2_assign_atomid_new(position_array, id1, id2, id3)
    normal_updated = np.empty(normal.shape)
    for frame in range(len(position_array)):
        if np.dot(normal[frame], normal2[frame]) < 0:
            normal_updated[frame] = -normal[frame]
        else:
            normal_updated[frame] = normal[frame]
    return np.stack((centers_overtraj, normal_updated), axis=1)


def normvector_connect_new(point1, point2, axis=1):
    """Return normalized vector from point2 to point1."""
    vec  = point1 - point2
    norm = np.sqrt(np.sum(vec ** 2, axis=axis, keepdims=True))
    return vec / norm


def angle_new(v1, v2, axis=1):
    """Angle (radians) between v1 and v2 with numerical stability clipping."""
    dot   = np.sum(v1 * v2, axis=axis)
    norm1 = np.sqrt(np.sum(v1 ** 2, axis=axis))
    norm2 = np.sqrt(np.sum(v2 ** 2, axis=axis))
    cos_a = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
    return np.arccos(cos_a)


def get_blockerrors_pyblock_nanskip(Data: np.ndarray, bound_frac: float):
    """Per-column mean and block error using the pyblock library.

    Falls back to zero error when pyblock is unavailable or the column is
    constant (average == 0 or 1).

    Parameters
    ----------
    Data : np.ndarray, shape (n_frames, n_columns)
    bound_frac : float
        Divisor applied to both average and error (use 1.0 for no scaling).

    Returns
    -------
    (ave, be) : two np.ndarrays of shape (n_columns,)
    """
    n_cols = Data.shape[1]
    ave, block_errors = [], []
    for i in range(n_cols):
        col     = Data[:, i]
        average = np.average(col)
        if pyblock is not None and average not in (0.0, 1.0):
            reblock_data = pyblock.blocking.reblock(col)
            opt = pyblock.blocking.find_optimal_block(len(col), reblock_data)[0]
            if math.isnan(opt):
                be = max(row[4] for row in reblock_data)
            else:
                be = reblock_data[opt][4]
        else:
            be = 0.0
        ave.append(average)
        block_errors.append(be)
    return np.asarray(ave) / bound_frac, np.asarray(block_errors) / bound_frac


def get_blockerror_pyblock_nanskip(data: np.ndarray):
    """Block error for a single 1-D array (scalar average + scalar error).

    Used internally by get_Kd(). Returns (average, block_error).
    """
    average = np.average(data)
    if pyblock is not None and average not in (0.0, 1.0):
        reblock_data = pyblock.blocking.reblock(data)
        opt = pyblock.blocking.find_optimal_block(len(data), reblock_data)[0]
        if math.isnan(opt):
            be = max(row[4] for row in reblock_data)
        else:
            be = reblock_data[opt][4]
    else:
        be = 0.0
    return average, float(be)


def dssp_convert(dssp: np.ndarray, secondary_motif: str,
                 get_over_time: bool = False,
                 weights=None) -> tuple:
    """Convert raw MDTraj DSSP strings to binary probabilities with block errors.

    Parameters
    ----------
    dssp : np.ndarray, shape (n_frames, n_residues)
        Raw DSSP array from md.compute_dssp (simplified=True).
        Values: 'H' helix, 'E' sheet, 'C' coil.
    secondary_motif : str
        'H' for helix or 'E' for sheet.
    get_over_time : bool
        When True, also compute per-frame fraction of residues in this motif
        (mean ± std across residues at each frame).
    weights : np.ndarray or str, optional
        Per-frame weights (1-D array) or path to a whitespace-delimited text
        file. Only applied when secondary_motif='H'.

    Returns
    -------
    (per_residue, per_frame, reweighted)
        per_residue : np.ndarray, shape (n_residues, 2) — [mean, block_error]
        per_frame   : np.ndarray, shape (n_frames, 2) or None — [mean, std] over residues
        reweighted  : list[float] — reweighted per-residue propensity (empty when no weights)
    """
    import time as _time
    t0 = _time.time()

    if secondary_motif not in {'H', 'E'}:
        raise ValueError("secondary_motif must be 'H' or 'E'")

    dssp_binary = (dssp == secondary_motif).astype(np.float32)  # (n_frames, n_residues)

    # Per-residue: mean + block error over the frames axis
    ave, be = get_blockerrors_pyblock_nanskip(dssp_binary, 1.0)
    per_residue = np.column_stack([ave, be])  # (n_residues, 2)

    # Optional reweighting (helix only)
    reweighted: list = []
    if weights is not None and secondary_motif == 'H':
        if isinstance(weights, str):
            weights = np.loadtxt(weights)
        n_fr = dssp_binary.shape[0]
        w = weights[: n_fr - 1] if weights.shape[0] == n_fr - 1 else weights[:n_fr]
        for i in range(dssp_binary.shape[1]):
            col = dssp_binary[: len(w), i]
            reweighted.append(float(np.dot(col, w)))

    # Per-frame: fraction of residues in this motif (mean ± std over residues per frame)
    per_frame = None
    if get_over_time:
        per_frame = np.column_stack([
            dssp_binary.mean(axis=1),
            dssp_binary.std(axis=1),
        ])  # (n_frames, 2)

    print(f'\tDSSP {secondary_motif} done in {round(_time.time() - t0, 2)} s')
    return per_residue, per_frame, reweighted


def Kd_calc(bound: float, conc: float) -> float:
    """Dissociation constant from bound fraction and ligand concentration (M)."""
    return (1 - bound) * conc / bound


def get_Kd(prot_lig_traj, simulation_times, contact_overTime: np.ndarray):
    """Compute Kd (mM) and bound-fraction time series from a binary contact vector.

    Parameters
    ----------
    prot_lig_traj : mdtraj.Trajectory
        Protein+ligand trajectory; used for box size (unitcell_lengths).
    simulation_times : list
        Frame times in ps (from self.simulation_times).
    contact_overTime : np.ndarray, shape (n_frames,)
        Per-frame total contact count; binarised internally.

    Returns
    -------
    kd : tuple (float, float)
        (KD in mM, KD_error in mM).
    bound_fraction_df : pd.DataFrame
        Cumulative bound-fraction time series indexed by frame time.
    bound_fraction : tuple (float, float)
        (boundfrac, boundfrac_be) over the full trajectory.
    """
    box_L = prot_lig_traj.unitcell_lengths[0][0]          # nm
    box_V_L = ((box_L * 1e-9) ** 3) * 1000                # litres
    concentration = 1.0 / (box_V_L * 6.023e23)            # mol/L

    contact_binary = (contact_overTime > 0).astype(int)
    boundfrac, boundfrac_be = get_blockerror_pyblock_nanskip(contact_binary)

    kd_val   = Kd_calc(boundfrac,              concentration) * 1000   # mM
    kd_upper = Kd_calc(boundfrac + boundfrac_be, concentration) * 1000
    kd_error = kd_val - kd_upper

    time_axis = np.linspace(0, simulation_times[-1], len(contact_binary))
    stride = 1
    bf_by_frame, err_up, err_low, t2 = [], [], [], []
    for i in tqdm(range(stride, len(contact_binary), stride)):
        bf, be = get_blockerror_pyblock_nanskip(contact_binary[:i])
        bf_by_frame.append(bf)
        err_up.append(bf - be)
        err_low.append(bf + be)
        t2.append(time_axis[i])

    bound_fraction_df = pd.DataFrame({
        'bound_fraction':       bf_by_frame,
        'bound_fraction_error_up':  err_up,
        'bound_fraction_error_low': err_low,
    }, index=t2)

    return (kd_val, kd_error), bound_fraction_df, (boundfrac, boundfrac_be)


# ── H-bond helpers ───────────────────────────────────────────────────────────

def _get_bond_triplets_print(topology, lig_donors, exclude_water=True,
                              sidechain_only=False, offset=0):
    """Return a dict of donor/acceptor information for the topology.

    Used by print_donors_acceptors() to build a human-readable inventory of
    N–H, O–H and S–H donors plus O/N/S acceptors.
    """
    hbond_donors_acceptors_dict = {}

    def can_participate(atom):
        if exclude_water and atom.residue.is_water:
            return False
        if sidechain_only and not atom.is_sidechain:
            return False
        return True

    def get_donors(e0, e1):
        elems = set((e0, e1))
        atoms = [(one, two) for one, two in topology.bonds
                 if set((one.element.symbol, two.element.symbol)) == elems]
        atoms = [pair for pair in atoms
                 if can_participate(pair[0]) and can_participate(pair[1])]
        indices = []
        for a0, a1 in atoms:
            pair = (a0.index, a1.index)
            if a0.element.symbol == e1:
                pair = pair[::-1]
            indices.append(pair)
        return indices

    nbonds = 0
    for _ in topology.bonds:
        nbonds += 1
        break
    if nbonds == 0:
        raise ValueError('No bonds found in topology. Try '
                         'traj._topology.create_standard_bonds().')

    def _fmt(idx):
        a = topology.atom(idx)
        return f'{a.residue.name} {a.residue.resSeq + offset} {a.name}'

    hbond_donors_acceptors_dict['NH_donors'] = [
        f'{_fmt(i[0])} -> {topology.atom(i[1]).name}' for i in get_donors('N', 'H')
    ]
    hbond_donors_acceptors_dict['OH_donors'] = [
        f'{_fmt(i[0])} -> {topology.atom(i[1]).name}' for i in get_donors('O', 'H')
    ]
    hbond_donors_acceptors_dict['SH_donors'] = [
        f'{_fmt(i[0])} -> {topology.atom(i[1]).name}' for i in get_donors('S', 'H')
    ]

    acceptor_elements = frozenset(('O', 'N', 'S'))
    hbond_donors_acceptors_dict['acceptors'] = [
        f'{topology.atom(a.index).residue.name} '
        f'{topology.atom(a.index).residue.resSeq + offset} {a.name}'
        for a in topology.atoms
        if a.element.symbol in acceptor_elements and can_participate(a)
    ]
    return hbond_donors_acceptors_dict


def baker_hubbard2(traj, freq=0.1, exclude_water=True, periodic=True,
                   sidechain_only=False, distance_cutoff=0.35, angle_cutoff=150,
                   lig_donor_index=[]):
    """Per-frame H-bond detection using Baker–Hubbard geometry criteria.

    Extended version of MDTraj's baker_hubbard that accepts additional ligand
    donor pairs via lig_donor_index.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Single-frame trajectory.
    freq : float
        Minimum fraction of frames a triplet must appear in.
    distance_cutoff : float
        Donor–acceptor distance cutoff in nm.
    angle_cutoff : float
        Donor–H–acceptor angle cutoff in degrees.
    lig_donor_index : list of [int, int]
        Additional donor–hydrogen index pairs from the ligand
        (MDTraj atom indices in the prot_lig topology context).

    Returns
    -------
    bond_triplets : np.ndarray, shape (n_hbonds, 3)
        [donor, hydrogen, acceptor] indices for detected H-bonds.
    raw : tuple (bond_triplets_raw, distances_raw)
        Full potential-triplet array and raw distances (for save_pca path).
    """
    angle_cutoff_rad = np.radians(angle_cutoff)

    if traj.topology is None:
        raise ValueError('baker_hubbard2 requires topology information')

    # Standard MDTraj triplets (protein only — no lig_donors parameter)
    bond_triplets = _get_bond_triplets(
        traj.topology,
        exclude_water=exclude_water,
        sidechain_only=sidechain_only,
    )

    # Append ligand donor triplets: pair each lig (heavy, H) with every O/N/S acceptor
    if lig_donor_index:
        acceptor_elements = frozenset(('O', 'N', 'S'))
        acceptors = [a.index for a in traj.topology.atoms
                     if a.element.symbol in acceptor_elements]
        lig_triplets = np.array(
            [(heavy, h, acc)
             for heavy, h in lig_donor_index
             for acc in acceptors
             if heavy != acc],
            dtype=int,
        )
        if lig_triplets.size:
            bond_triplets = np.vstack([bond_triplets, lig_triplets])

    bond_triplets_raw = bond_triplets.copy()

    result = _compute_bounded_geometry(
        traj, bond_triplets, distance_cutoff, [1, 2], [0, 1, 2],
        freq=freq, periodic=periodic,
    )
    if len(result) == 4:
        mask, distances, angles, distances_raw = result
    else:
        mask, distances, angles = result
        distances_raw = distances  # older MDTraj — no raw distances returned
    presence    = np.logical_and(distances < distance_cutoff, angles > angle_cutoff_rad)
    mask[mask]  = np.mean(presence, axis=0) > freq

    return bond_triplets.compress(mask, axis=0), (bond_triplets_raw, distances_raw)


def print_donors_acceptors(traj, exclude_water=True, sidechain_only=False,
                           angle_cutoff=150, lig_donor_index=[], offset=0):
    """Return a topology-level inventory of H-bond donors and acceptors.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        Single-frame trajectory (only topology is used).
    lig_donor_index : list of (str, str)
        Ligand donor atom name pairs (passed through for reference; topology
        donor scan uses topology bonds only).
    offset : int
        Residue number offset applied to labels.

    Returns
    -------
    dict with keys 'NH_donors', 'OH_donors', 'SH_donors', 'acceptors'.
    """
    return _get_bond_triplets_print(
        traj.topology,
        lig_donors=lig_donor_index,
        exclude_water=exclude_water,
        sidechain_only=sidechain_only,
        offset=offset,
    )


def add_hbond_pair(donor, acceptor, hbond_pairs, donor_res):
    """Accumulate an H-bond observation into the pairs dictionary.

    hbond_pairs[donor_res][donor][acceptor] counts the number of frames in
    which this donor–acceptor interaction was observed.
    """
    if donor_res not in hbond_pairs:
        hbond_pairs[donor_res] = {}
    if donor not in hbond_pairs[donor_res]:
        hbond_pairs[donor_res][donor] = {}
    if acceptor not in hbond_pairs[donor_res][donor]:
        hbond_pairs[donor_res][donor][acceptor] = 0
    hbond_pairs[donor_res][donor][acceptor] += 1


# ── MRC volume writer ────────────────────────────────────────────────────────

def _write_mrc_field(field: np.ndarray, xmin, dx: float, fname: str,
                     gaussian_sigma: float = None) -> None:
    """Write a 3D scalar field as an MRC volume file for PyMOL visualization.

    Parameters
    ----------
    field : np.ndarray, shape (nx, ny, nz)
        3D array of scalar values.
    xmin : array-like, shape (3,)
        Physical coordinates (Å) of the grid origin.
    dx : float
        Voxel edge length in Å.
    fname : str
        Output path (.mrc).
    gaussian_sigma : float, optional
        Gaussian smoothing in voxels applied before writing.
        Typical range: 0.5 (subtle) to 2.0 (heavy). None = no smoothing.
    """
    if not MRCFILE_AVAILABLE:
        raise ImportError("mrcfile required — install: pip install mrcfile")
    data = np.asarray(field, dtype=np.float32)
    if gaussian_sigma is not None and gaussian_sigma > 0:
        if _scipy_ndimage is not None:
            data = _scipy_ndimage.gaussian_filter(
                data, sigma=gaussian_sigma).astype(np.float32)
        else:
            print(f"Warning: scipy not available — Gaussian smoothing skipped for {fname}")
    with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(data.T)   # MRC convention: ZYX order
        mrc.voxel_size = dx
        mrc._set_nstart(0, 0, 0)
        mrc.header.origin.flags.writeable = True
        mrc.header.origin.x = float(xmin[0])
        mrc.header.origin.y = float(xmin[1])
        mrc.header.origin.z = float(xmin[2])
        mrc.update_header_from_data()
        mrc.update_header_stats()


def _jet_hex(t: float) -> str:
    """Map a normalised value in [0, 1] to a Jet colormap hex string.

    Jet anchors (darkblue → blue → cyan → yellow → red):
        0.000 → #000080   0.125 → #0000ff   0.375 → #00ffff
        0.625 → #ffff00   0.875 → #ff0000   1.000 → #800000
    Returns a 6-character lowercase hex string without the leading '#'.
    """
    anchors = [
        (0.000, 0.0, 0.0, 0.5),
        (0.125, 0.0, 0.0, 1.0),
        (0.375, 0.0, 1.0, 1.0),
        (0.625, 1.0, 1.0, 0.0),
        (0.875, 1.0, 0.0, 0.0),
        (1.000, 0.5, 0.0, 0.0),
    ]
    v = float(np.clip(t, 0.0, 1.0))
    for i in range(len(anchors) - 1):
        t0, r0, g0, b0 = anchors[i]
        t1, r1, g1, b1 = anchors[i + 1]
        if v <= t1 + 1e-9:
            f = (v - t0) / (t1 - t0) if (t1 - t0) > 0 else 0.0
            r = int((r0 + f * (r1 - r0)) * 255)
            g = int((g0 + f * (g1 - g0)) * 255)
            b = int((b0 + f * (b1 - b0)) * 255)
            return f"{r:02x}{g:02x}{b:02x}"
    r, g, b = (int(v * 255) for v in anchors[-1][1:])
    return f"{r:02x}{g:02x}{b:02x}"


# ── PCA and trajectory clustering helpers ────────────────────────────────────

def res_space(nres: int, space: int) -> np.ndarray:
    """Residue pair index filter: excludes neighbours within ±space positions."""
    arr = np.arange(nres)
    return np.hstack([arr[np.abs(arr - i) > space] + i * nres for i in range(nres)])


def _pca_center(x: np.ndarray) -> np.ndarray:
    return x - x.mean(0)


def _pca_cov(x: np.ndarray, y: np.ndarray = None) -> np.ndarray:
    if y is None:
        y = x
    return _pca_center(x).T @ _pca_center(y) * (1 / (len(x) - 1))


def pca(x: np.ndarray, dim: int):
    """PCA via eigenvalue decomposition of the covariance matrix.

    Parameters
    ----------
    x   : np.ndarray, shape (n_samples, n_features)
    dim : int — number of principal components to retain

    Returns
    -------
    projection   : np.ndarray, shape (n_samples, dim)
    eigenvalues  : np.ndarray, shape (dim,)  — sorted descending
    eigenvectors : np.ndarray, shape (n_features, dim)
    """
    l, v = np.linalg.eigh(_pca_cov(x))
    idx  = l.argsort()[::-1]
    v, l = v[..., idx], l[idx]
    l, v = l[:dim], v[..., :dim]
    return x @ v, l, v


def fes2d(x, y, ax=None, xlabel=None, ylabel=None, cmap='jet', cbar=True,
          bins=50, weights=None, vmin=None, vmax=None,
          scatterx=None, scattery=None):
    """2D free energy surface as −log(P) from a 2D histogram.

    Returns (contour_set_or_None, fes_df).
    fes_df is a DataFrame indexed by y-bin centres with x-bin centres as columns.
    The contour plot is skipped when matplotlib is not available.
    """
    z, xe, ye = np.histogram2d(x, y, bins=bins, weights=weights)
    extent     = (xe.min(), xe.max(), ye.min(), ye.max())
    arr        = np.ma.masked_array(z, z == 0)
    F          = -np.log(arr)
    F         += -F.min()

    fes_df = pd.DataFrame(
        F.T,
        index=[float(v.real) for v in ye[:-1]],
        columns=[float(v.real) for v in xe[:-1]],
    )

    if _plt is None:
        return None, fes_df

    if ax is None:
        _, ax = _plt.subplots(1, 1, sharex=True, sharey=True)
    if scatterx is not None and scattery is not None:
        ax.scatter(scatterx, scattery, c='grey', s=60, edgecolors='black', alpha=0.5)
    a = ax.contourf(F.T, 30, cmap=cmap, extent=extent, zorder=-1, vmin=vmin, vmax=vmax)
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) * 1.0)
    if cbar:
        cb = _plt.colorbar(a, ax=ax, fraction=0.046, pad=0.04, format='%.2f')
        cb.set_label('Free Energy / (kT)', size=25, labelpad=20)
        cb.ax.tick_params(labelsize=20)
    return a, fes_df


def kmeans(p: np.ndarray, k: int):
    """K-means clustering via deeptime.

    Returns (dtraj, frames_cl, cluster_centers).
    """
    if _DeeptimeKMeans is None:
        raise ImportError("deeptime required — install: pip install deeptime")
    cluster   = _DeeptimeKMeans(k, max_iter=1000).fit_fetch(p)
    dtraj     = cluster.transform(p)
    frames_cl = [np.where(dtraj == i)[0] for i in range(k)]
    return dtraj, frames_cl, cluster.cluster_centers


def plot_pca_clusters(projection: np.ndarray, dtraj: np.ndarray,
                      clustercenters: np.ndarray, title: str = '',
                      xlabel: str = 'PC1', ylabel: str = 'PC2'):
    """Plotly scatter of PCA projection coloured by cluster assignment.

    Returns go.Figure or None when Plotly is not available.
    """
    if not PLOTLY_AVAILABLE:
        return None
    n_cl   = int(dtraj.max()) + 1
    colors = [f'hsl({int(360 * i / n_cl)},70%,50%)' for i in range(n_cl)]
    fig    = go.Figure()
    for i in range(n_cl):
        mask = dtraj == i
        fig.add_trace(go.Scatter(
            x=projection[mask, 0], y=projection[mask, 1],
            mode='markers', name=f'Cluster {i + 1}',
            marker=dict(size=3, color=colors[i], opacity=0.6),
        ))
    for i, c in enumerate(clustercenters):
        fig.add_annotation(x=float(c[0]), y=float(c[1]), text=str(i + 1),
                           font=dict(size=14, color='white'), showarrow=False)
    fig.update_layout(title=title, xaxis_title=xlabel, yaxis_title=ylabel,
                      width=700, height=500)
    return fig


def plot_pca_pie(cluster_populations: 'pd.DataFrame'):
    """Plotly pie chart of PCA cluster population percentages.

    Returns go.Figure or None when Plotly is not available.
    """
    if not PLOTLY_AVAILABLE:
        return None
    fig = go.Figure(go.Pie(
        labels=[f'Cluster {int(c)}' for c in cluster_populations['Cluster']],
        values=cluster_populations['Percentage'],
        textinfo='label+percent',
    ))
    fig.update_layout(title='Cluster populations', width=500, height=400)
    return fig


def compute_pca(feature_matrix: np.ndarray, pca_dim: int = 2,
                analysis_dim: int = 2, n_clusters: int = 2,
                standard_scaler: bool = False):
    """PCA + K-means clustering pipeline.

    Mean-centering is performed internally by the covariance calculation.
    An optional StandardScaler step (zero mean, unit variance per feature)
    can be enabled with ``standard_scaler=True`` — this was the behaviour of
    the original REST-Analysis code and is needed to reproduce those results.
    Without it, features with higher post-kernel variance dominate the PCA.

    Parameters
    ----------
    feature_matrix : np.ndarray, shape (n_frames, n_features)
        Pre-processed feature matrix. For distance inputs with Gaussian kernel
        this is in [0,1]; otherwise raw distances in nm.
    pca_dim        : int — total principal components to compute
    analysis_dim   : int — PC dimensions used for K-means
    n_clusters     : int — number of K-means clusters
    standard_scaler : bool
        Apply sklearn StandardScaler (zero mean, unit std per feature) before
        PCA. Set to True to match the original REST-Analysis behaviour.
        Default False.

    Returns
    -------
    pca_result         : (projection, eigenvalues, eigenvectors)
    fes_df             : pd.DataFrame — 2D FES on PC1/PC2 (custom PCA)
    fes_skl_df         : pd.DataFrame — 2D FES from sklearn PCA (diagnostic)
    dtraj              : np.ndarray   — cluster assignment per frame
    frames_cl          : list[np.ndarray] — frame indices per cluster
    clustercenters     : np.ndarray   — cluster centroids in PC space
    clusters_fig       : go.Figure or None
    cluster_populations: pd.DataFrame
    silhouette         : float
    """
    if _sklearn_PCA is None:
        raise ImportError("scikit-learn required — install: pip install scikit-learn")
    if _DeeptimeKMeans is None:
        raise ImportError("deeptime required — install: pip install deeptime")

    print("Computing PCA …")

    # Optional StandardScaler — reproduces original REST-Analysis behaviour
    if standard_scaler:
        if _sklearn_StandardScaler is None:
            raise ImportError("scikit-learn required — install: pip install scikit-learn")
        print("  Applying StandardScaler (zero mean, unit std per feature) …")
        feature_matrix = _sklearn_StandardScaler().fit_transform(feature_matrix)

    # Sklearn PCA — diagnostic FES for comparison with the custom implementation
    pca_sk   = _sklearn_PCA(n_components=min(2, feature_matrix.shape[1]))
    proj_skl = pca_sk.fit_transform(feature_matrix)
    _, fes_skl_df = fes2d(proj_skl[:, 0], proj_skl[:, 1], cbar=False)

    # Custom PCA via eigendecomposition of the covariance matrix
    projection, l, v = pca(feature_matrix, dim=pca_dim)
    del feature_matrix
    print(f"  Eigenvalues: {l.shape}  Projection: {projection.shape}")

    _, fes_df = fes2d(projection[:, 0], projection[:, 1], cbar=False)

    # K-means on first analysis_dim PCs
    dtraj, frames_cl, clustercenters = kmeans(projection[:, :analysis_dim], n_clusters)
    silhouette = _sklearn_silhouette_score(projection, dtraj)

    clusters_fig = plot_pca_clusters(
        projection, dtraj, clustercenters,
        title=f'PCA clustering — silhouette: {silhouette:.3f}',
    )

    unique_vals, counts = np.unique(dtraj, return_counts=True)
    cluster_populations = pd.DataFrame({
        'Cluster':    unique_vals.astype(int).tolist(),
        'Count':      counts.astype(int).tolist(),
        'Percentage': [c / counts.sum() * 100 for c in counts],
    }).sort_values('Cluster').reset_index(drop=True)

    return (
        (projection, l, v),
        fes_df, fes_skl_df,
        dtraj, frames_cl, clustercenters,
        clusters_fig, cluster_populations, silhouette,
    )


# ── Graph clustering helpers ──────────────────────────────────────────────────
# Residue contact-network sub-clustering: builds one residue-residue contact
# graph per frame, measures Jaccard distance between frames' edge sets, and
# clusters frames by that distance. Intended to run on a PCA cluster subset
# (see compute_pca_trajectory / compute_pca above) as a second-level
# "graph on PCA subsets" clustering step.

def _valid_graph_clusters(labels: np.ndarray, allow_noise: bool = True) -> Dict[int, List[int]]:
    """Return dict cluster_label -> indices. Drops noise (-1) when allow_noise=False."""
    cl2idx = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab == -1 and not allow_noise:
            continue
        cl2idx[lab].append(i)
    return {c: idx for c, idx in cl2idx.items() if len(idx) > 0}


def silhouette_from_distance(D: np.ndarray, labels: np.ndarray) -> Tuple[Optional[float], Optional[np.ndarray]]:
    """Mean silhouette score from a precomputed distance matrix.

    Requires at least 2 clusters with no singletons (noise label -1 and
    singleton clusters are dropped first). Returns (mean_score, per_sample_scores)
    or (None, None) when fewer than 2 valid clusters remain.
    """
    if _sklearn_silhouette_score is None or _sklearn_silhouette_samples is None:
        raise ImportError("scikit-learn required — install: pip install scikit-learn")

    cl2idx = _valid_graph_clusters(labels, allow_noise=False)
    cl2idx = {c: idx for c, idx in cl2idx.items() if len(idx) >= 2}
    if len(cl2idx) < 2:
        return None, None
    keep = sorted([i for idx in cl2idx.values() for i in idx])
    D_sub = D[np.ix_(keep, keep)]
    y_sub = labels[keep]
    try:
        s   = _sklearn_silhouette_score(D_sub, y_sub, metric='precomputed')
        s_i = _sklearn_silhouette_samples(D_sub, y_sub, metric='precomputed')
        return float(s), s_i
    except Exception:
        return None, None


def _build_nx_graphs(top, frame_edge_sets: List[Set[Tuple[int, int]]]) -> list:
    """One networkx.Graph per frame: nodes = all residues, edges = contacts in that frame."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required — install: pip install networkx")
    node_ids = [r.index for r in top.residues]
    graphs = []
    for edges in frame_edge_sets:
        G = nx.Graph()
        G.add_nodes_from(node_ids)
        G.add_edges_from(edges)
        graphs.append(G)
    return graphs


def jaccard_distance_matrix(edge_sets: List[Set[Tuple[int, int]]]) -> np.ndarray:
    """Full pairwise Jaccard distance matrix between per-frame contact edge sets.

    distance = 1 - |A ∩ B| / |A ∪ B|

    Exact, O(n_frames²) pure-Python loop over set intersection/union — no
    vectorized or approximate (MinHash) fallback. Fine for a few thousand
    frames; can take a long time (potentially hours) at tens of thousands of
    frames. See DEVELOPMENT.md "Known limitation" for context.
    """
    print("Computing Jaccard distance matrix (exact)...")
    n = len(edge_sets)
    D = np.zeros((n, n), dtype=float)
    for i in tqdm(range(n)):
        Ai = edge_sets[i]
        for j in range(i + 1, n):
            Aj = edge_sets[j]
            if not Ai and not Aj:
                dist = 0.0
            else:
                inter = len(Ai & Aj)
                union = len(Ai | Aj)
                dist = 1.0 - (inter / union if union > 0 else 0.0)
            D[i, j] = D[j, i] = dist
    return D


def _cluster_by_distance(D: np.ndarray, clustering_method: str, n_clusters: Optional[int],
                         linkage: str, distance_threshold: Optional[float],
                         hdbscan_min_cluster_size: int = 10) -> np.ndarray:
    """Cluster frames from a precomputed distance matrix (agglomerative or hdbscan)."""
    if clustering_method == "hdbscan":
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan required for clustering_method='hdbscan' — "
                              "install: pip install hdbscan")
        clusterer = _hdbscan.HDBSCAN(metric='precomputed', min_cluster_size=hdbscan_min_cluster_size)
        return clusterer.fit_predict(D)

    if _sklearn_AgglomerativeClustering is None:
        raise ImportError("scikit-learn required — install: pip install scikit-learn")
    model = _sklearn_AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage=linkage,
        distance_threshold=None if n_clusters is not None else distance_threshold,
    )
    return model.fit_predict(D)


def _representative_frames(graphs: list, labels: np.ndarray,
                           criterion: str = "max_degree") -> Dict[int, int]:
    """Pick one representative frame per cluster.

    criterion='max_degree' picks the frame with the highest mean node degree
    (densest contact network); any other value falls back to the frame with
    the most edges. Noise label (-1, HDBSCAN only) is skipped.
    """
    reps: Dict[int, int] = {}
    for cl in sorted(set(labels)):
        if cl == -1:
            continue
        idx = np.where(labels == cl)[0]
        if criterion == "max_degree":
            best_i, best_score = None, -1
            for i in idx:
                degs = [d for _, d in graphs[i].degree()]
                score = float(np.mean(degs)) if degs else 0.0
                if score > best_score:
                    best_i, best_score = i, score
            reps[cl] = int(best_i)
        else:
            best_i = max(idx, key=lambda i: graphs[i].number_of_edges())
            reps[cl] = int(best_i)
    return reps


def _save_gexf(graphs: list, out_dir: str) -> None:
    """Write one .gexf file per frame graph (viewable in Gephi/Cytoscape)."""
    if not NETWORKX_AVAILABLE:
        raise ImportError("networkx required — install: pip install networkx")
    os.makedirs(out_dir, exist_ok=True)
    for i, G in enumerate(graphs):
        nx.write_gexf(G, os.path.join(out_dir, f"frame_{i:06d}.gexf"))


# ════════════════════════════════════════════════════════════════════════════
# PharmacophoreTrajectory
# ════════════════════════════════════════════════════════════════════════════

class PharmacophoreTrajectory:
    """Load and analyse MD trajectories for pharmacophore definition.

    Hybrid approach:
      - MDAnalysis + RDKit for ligand extraction and chemical feature typing
      - MDTraj for trajectory loading and all distance/contact computation

    Example
    -------
    >>> sim = PharmacophoreTrajectory("system.gro", "traj.xtc")
    >>> sim.load(stride=1)                     # ligand auto-detected
    >>> sim.load(ligand_resname="LIG")         # explicit override
    >>> rings = sim.get_ligand_rings()
    >>> img, mol = pu.draw_molecule_with_labels(sim.ligand_mol)
    """

    def __init__(self, topology: str, trajectory: str) -> None:
        self._topology_path   = topology
        self._trajectory_path = trajectory

        # Populated by load()
        self.traj          = None   # full MDTraj trajectory
        self.protein_traj  = None   # protein-only slice
        self.protein_top   = None
        self.prot_lig_traj = None   # protein + ligand slice
        self.prot_lig_top  = None
        self.ligand_traj   = None   # ligand-only slice
        self.ligand_top    = None
        self.ligand_resname = None
        self.ligand_mol    = None   # RDKit Mol
        self.offset        = 0

        # Populated by _build_topology_definitions()
        self.residue_names            = None
        self.residue_names_dict       = None
        self.chain_counter            = 0
        self.simulation_times         = None
        self.protein_residue_numbers  = None
        self.ligand_residue_numbers   = None
        self.hydrophobic_residue_atoms_dict = None

        # Selection indices in prot_lig context (used for hbond computation)
        self._ligand_sel_idx  = None
        self._protein_sel_idx = None

        # Populated by get_ligand_rings(); used by get_atom_property()
        # NOTE: these are atom names from the RDKit PDB — validate against
        #       MDTraj atom names during benchmarking.
        self._ligand_aromatic_atom_names: set = set()
        # When not None, get_ligand_rings() returns this instead of running RDKit.
        # Each entry needs at minimum: {'atom_names': [...], 'aromatic': bool}
        self._manual_rings: Optional[List[Dict]] = None

        # Populated by compute_aromatic_contacts(); used by compute_contact_probability()
        self._stacked_full        = None   # (n_frames, n_protein_residues) geometric (p+t)
        self._pstacked_full       = None
        self._tstacked_full       = None
        self._stacked_by_ring     = None   # list[(n_frames, n_protein_residues)] per ligand ring
        self._aromatic_rings      = None   # ring dicts from get_ligand_rings()
        self._protein_rings_index = None   # residue indices of aromatic protein residues
        self.stacking_contacts_dict = None  # {'sum': array, '0': array, ...} for pharmacophore

        # Populated by compute_aromatic_contacts()
        self.aromatic_contact_probability = None  # pd.DataFrame indexed by residue name

        # Populated by compute_contact_probability()
        self.contact_probability  = None   # pd.DataFrame indexed by residue name
        self.dual_contact_matrix  = None   # (n_residues, n_residues) co-occurrence DataFrame
        self.kd                   = None   # (KD_mM, KD_error_mM)
        self.bound_fraction       = None   # (boundfrac, boundfrac_be)
        self.kd_over_time         = None   # pd.DataFrame cumulative bound-fraction time series

        # Populated by compute_hydrophobic_contacts()
        self.hydrophobic_contact_probability      = None  # pd.DataFrame indexed by residue name
        self.hydrophobic_contact_frames           = None  # (n_frames, n_pairs) binary array
        self.hydrophobic_atom_contact_probability = None  # ligand atom × protein atom DataFrame
        self.hydrophobic_ligand_atom_probability  = None  # per-ligand-atom probability DataFrame
        self.hydrophobic_distances_df             = None  # per-frame per-residue mean distance (if save_pca=True)

        # Populated by compute_hbond_contacts()
        self.hbond_contact_probability     = None  # pd.DataFrame indexed by residue name
        self.hbond_contact_frames_pd       = None  # (n_frames, n_residues) protein-donor binary
        self.hbond_contact_frames_ld       = None  # (n_frames, n_residues) ligand-donor binary
        self.hbond_pairs_pd                = None  # {res_idx: {prot_donor: {lig_acc: count}}}
        self.hbond_pairs_ld                = None  # {res_idx: {lig_donor: {prot_acc: count}}}
        self.hbond_donors_acceptors        = None  # topology-level donor/acceptor inventory
        self.hbond_ligand_atom_probability = None  # per-ligand-atom LD/PD probability DataFrame
        self.hbond_distances_df            = None  # per-frame raw distances (if save_pca=True)
        self.hbond_residue_distances_df    = None  # residue-pair distances (if save_pca=True)

        # Populated by compute_dssp()
        self.dssp_df           = None  # pd.DataFrame indexed by residue name
        self.dssp_over_time_df = None  # pd.DataFrame indexed by simulation time (if get_over_time=True)

        # Populated by compute_gyration_salpha()
        self.rg_over_time            = None  # np.ndarray, shape (n_frames,), Rg in nm
        self.salpha_over_time        = None  # np.ndarray, shape (n_frames,), per-frame total Sα
        self.gyration_salpha_fes_df  = None  # pd.DataFrame, 2D FES (Sα rows × Rg cols, kcal/mol)

        # Populated by compute_contact_map()
        self.contact_map_df     = None  # (n_res × n_res) contact probability DataFrame
        self.contact_map_raw_df = None  # (n_frames × n_pairs) raw distance DataFrame

        # Populated by compute_sasa()
        self.sasa_atoms_df         = None  # (n_frames × n_protein_atoms)    SASA in nm²
        self.sasa_residues_df      = None  # (n_frames × n_protein_residues) SASA in nm²
        self.sasa_ligand_atoms_df  = None  # (n_frames × n_ligand_atoms)     SASA in nm²
                                           # computed on prot_lig_traj (protein context)

        # Populated by compute_residue_dot_positions()
        self.residue_dot_positions = None  # dict: 'aromatic'/'hydrophobic'/'hba'/'hbd'
                                           # → np.ndarray (N, 3) absolute Å coordinates
                                           # (same system as MRC voxels — no centroid shift)

        # Populated by compute_all_atom_contacts()
        self.all_atom_contact_probability     = None  # pd.DataFrame indexed by residue name
        self.all_atom_contact_frames          = None  # (n_frames, n_lig_heavy * n_prot_heavy) binary
        self.all_atom_ligand_atom_probability = None  # per-ligand-atom contact probability DataFrame
        self.all_atom_distances_df            = None  # (n_frames, n_prot_heavy) mean distances (if save_pca=True)

        # Populated by compute_pca_trajectory()
        self.pca_result              = None   # (projection, eigenvalues, eigenvectors)
        self.pca_fes_df              = None   # 2D FES DataFrame (custom PCA)
        self.pca_fes_skl_df          = None   # 2D FES DataFrame (sklearn PCA, diagnostic)
        self.pca_dtraj               = None   # cluster assignment per frame, shape (n_frames,)
        self.pca_frames_cl           = None   # list of frame index arrays per cluster
        self.pca_cluster_centers     = None   # cluster centroids in PC space
        self.pca_clusters_fig        = None   # Plotly figure of PCA scatter
        self.pca_cluster_populations = None   # cluster size DataFrame
        self.pca_silhouette_score    = None   # float

        # Populated by compute_graph_clustering()
        self.graph_cluster_distance_matrix = None  # (n_frames, n_frames) Jaccard distance matrix
        self.graph_cluster_graphs          = None  # list[nx.Graph], one per frame
        self._graph_cluster_traj_attr      = None  # 'protein_traj' or 'prot_lig_traj' —
                                                    # which trajectory D/graphs are aligned to
        self._graph_cluster_full_n_frames  = None  # frame count of the original, un-sliced simulation —
                                                    # used by save_graph_cluster_trajectories()'s population filter

        # Populated by cluster_graph_clustering()
        self.graph_cluster_labels          = None  # (n_frames,) cluster label per frame (-1 = noise, hdbscan only)
        self.graph_cluster_df              = None  # pd.DataFrame: frame, time_ps, cluster, n_edges, mean_degree, Rg_nm
        self.graph_cluster_representatives = None  # dict: cluster_label -> representative frame index (local)
        self.graph_cluster_silhouette      = None  # float or None (None if <2 valid non-singleton clusters)

        # Populated by save_graph_cluster_trajectories()
        self.graph_cluster_saved_paths     = None  # dict: cluster_label -> {trajectory_path, structure_path,
                                                    #                          representative_frame, subset_frames}

        # Populated by compute_negative_space()
        self.negative_space_data          = None   # dict: masks, free_fraction, protein_occupancy, grid_info, volumes

        # Populated by define_pharmacophore()
        self.pharmacophore_maps_contested = None   # residue-type feature maps on contested space (occ ≥ 30%)
        self.pharmacophore_maps_full      = None   # residue-type feature maps on full shell (no threshold)
        self.growth_space_features        = None   # chemical features + linear scores for growth space

    # ── 1. Load ──────────────────────────────────────────────────────────────

    def load(self, ligand_resname: str = None, stride: int = 1, offset: int = 0) -> None:
        """Load topology + trajectory, extract ligand, build topology definitions.

        Parameters
        ----------
        ligand_resname : str, optional
            Ligand residue name. Auto-detected from topology when not provided.
        stride : int
            Load every N-th frame.
        offset : int
            Added to residue sequence numbers to match experimental convention.
            Defaults to 0; wire up CLI/notebook I/O when finalizing the algorithm.
        """
        # Step 1: MDAnalysis topology scan — auto-detect ligand
        topo_info = analyze_topology(self._topology_path)
        if 'error' in topo_info:
            print(f"Warning: topology scan failed — {topo_info['error']}")
            topo_info = {'has_ligand': False}

        if ligand_resname is not None:
            self.ligand_resname = ligand_resname
        elif topo_info.get('has_ligand'):
            self.ligand_resname = topo_info['ligand_resname']
            print(f"Auto-detected ligand: {self.ligand_resname}")
        else:
            self.ligand_resname = None

        # Step 2: load full trajectory (MDTraj)
        self.traj = md.load(self._trajectory_path, top=self._topology_path, stride=stride)
        print(f"\t{self._topology_path}")
        print(f"\t{self._trajectory_path}")
        print(f"\t{self.traj}")

        # Step 3: protein-only slice
        # Explicit exclusion prevents MDTraj from including the ligand in the protein slice
        protein_sel_str = (
            f'protein and not resname {self.ligand_resname}'
            if self.ligand_resname else 'protein'
        )
        protein_sel = self.traj.topology.select(protein_sel_str)
        self.protein_traj = self.traj.atom_slice(protein_sel)
        self.protein_top  = self.protein_traj.topology

        # Step 4: protein+ligand and ligand-only slices
        if self.ligand_resname:
            prot_lig_sel = self.traj.topology.select(
                f'protein or resname {self.ligand_resname}'
            )
            self.prot_lig_traj = self.traj.atom_slice(prot_lig_sel)
            self.prot_lig_top  = self.prot_lig_traj.topology

            lig_sel = self.traj.topology.select(f'resname {self.ligand_resname}')
            self.ligand_traj = self.traj.atom_slice(lig_sel)
            self.ligand_top  = self.ligand_traj.topology

            self._ligand_sel_idx  = self.prot_lig_top.select(f'resname {self.ligand_resname}')
            self._protein_sel_idx = self.prot_lig_top.select('protein')

        # Step 5: extract ligand as RDKit mol (MDAnalysis)
        if self.ligand_resname:
            self.ligand_mol = extract_ligand_mol(self._topology_path, self.ligand_resname)

        # Step 6: build residue/atom definitions
        self.offset = offset
        self._build_topology_definitions()

        print(f"\nLoaded: {self.traj.n_frames} frames")
        print(f"Protein residues: {self.protein_top.n_residues}")
        if self.ligand_resname:
            print(f"Ligand '{self.ligand_resname}': {self.ligand_top.n_atoms} atoms")

    def _build_topology_definitions(self) -> None:
        """Build residue numbering, atom selection arrays, and index dictionaries."""
        residue_names, chain_ids, residue_names_dict = [], [], {}
        chain_counter = residue_counter = 0

        for residue in self.protein_top.residues:
            name = f"{residue.name}_{residue.resSeq + self.offset}"
            if name in residue_names and residue_counter == len(residue_names):
                chain_counter += 1
                residue_counter = 0
            residue_names.append(name)
            chain_ids.append(chain_counter)
            residue_names_dict[residue.index] = name
            residue_counter += 1

        if chain_counter > 0:
            residue_names = [f"{n}_{c}" for n, c in zip(residue_names, chain_ids)]

        self.residue_names       = residue_names
        self.residue_names_dict  = residue_names_dict
        self.chain_counter       = chain_counter
        self.simulation_times    = self.protein_traj.time.tolist()
        self.protein_residue_numbers = [r.index for r in self.protein_top.residues]

        # Protein atom selections
        self.all_protein_atoms        = self.protein_top.select('all')
        self.all_protein_atoms_noh    = self.protein_top.select('all and not element H')
        self.hydrophobic_atoms_protein = self.protein_top.select('element C')

        # Hydrophobic residue → atom counter mapping
        # atom_counter is 0-based within the C-only selection (not global atom index)
        hydrophobic_residue_atoms = {}
        for atom_counter, atom_idx in enumerate(self.hydrophobic_atoms_protein):
            atom = self.protein_top.atom(atom_idx)
            key  = f"{atom.residue.name}_{atom.residue.resSeq + self.offset}"
            hydrophobic_residue_atoms.setdefault(key, []).append(atom_counter)
        self.hydrophobic_residue_atoms_dict = hydrophobic_residue_atoms

        if not self.ligand_resname:
            return

        # Ligand-specific definitions (all indices in prot_lig context)
        self.ligand_residue_numbers = [
            r.index for r in self.prot_lig_top.residues
            if r.name == self.ligand_resname
        ]
        self.all_ligand_atoms     = self.prot_lig_top.select(f'resname {self.ligand_resname}')
        self.all_ligand_atoms_noh = self.prot_lig_top.select(
            f'resname {self.ligand_resname} and not element H'
        )
        self.hydrophobic_atoms_ligand = self.prot_lig_top.select(
            f'resname {self.ligand_resname} and (element C or element S)'
        )

        # Combined protein+ligand residue name dict
        max_prot_key = max(residue_names_dict.keys()) if residue_names_dict else -1
        lig_residue_names_dict = {
            max_prot_key + 1 + i: f"{r.name}_{r.resSeq + self.offset}"
            for i, r in enumerate(self.ligand_top.residues)
        }
        self.all_residue_names_dict = residue_names_dict | lig_residue_names_dict

        # Atom name dict (prot_lig context — for contact matrix labeling)
        self.prot_lig_atom_names_dict = {
            atom.index: f'{atom.name}_{atom.residue.name}_{atom.residue.resSeq + self.offset}'
            for atom in self.prot_lig_top.atoms
        }

        # Ordered ligand atom labels for dataframe indexing
        self.ligand_atom_labels = [
            f'{self.prot_lig_top.atom(n).name}_{self.prot_lig_top.atom(n).residue.name}'
            f'_{self.prot_lig_top.atom(n).residue.resSeq + self.offset}'
            for n in self.prot_lig_top.select(f'resname {self.ligand_resname}')
        ]

        # Hydrophobic contact pairs (ligand C/S × protein C)
        self.protein_ligand_hphob_pairs = np.array(
            list(product(self.hydrophobic_atoms_ligand, self.hydrophobic_atoms_protein))
        )

    # ── 1b. Trajectory alignment ──────────────────────────────────────────────

    def ligand_align(self, selection: str) -> None:
        """Superpose the trajectory on a subset of ligand atoms, in-place.

        Centers and aligns every frame of ``prot_lig_traj`` onto the first
        frame using the atoms identified by *selection*.  The operation is
        in-place (MDTraj ``superpose`` returns ``self``), so no extra RAM is
        consumed and no file is written to disk.

        Call this immediately after ``load()`` and before any contact or
        voxel computation.

        Parameters
        ----------
        selection : str
            MDTraj selection string identifying the reference atoms, e.g.
            ``"resname EPI and (name O1 or name C11 or name C18)"``.
            Use ``draw_molecule_with_labels()`` to identify atom names.
        """
        if self.prot_lig_traj is None:
            raise RuntimeError("Call load() before ligand_align()")

        atom_indices = self.prot_lig_top.select(selection)
        if len(atom_indices) == 0:
            raise ValueError(f"Selection '{selection}' matched no atoms in prot_lig_top")

        print(f"Aligning on {len(atom_indices)} atom(s): {selection}")
        self.prot_lig_traj.superpose(
            reference=self.prot_lig_traj,
            frame=0,
            atom_indices=atom_indices,
        )

        # Sync ligand_traj to the aligned coordinates so ligand_centroid.pdb
        # (written by save_mrc_files from ligand_traj) matches the MRC grid.
        if self.ligand_traj is not None:
            self.ligand_traj.xyz[:] = self.prot_lig_traj.xyz[:, self._ligand_sel_idx, :]

        print("Alignment done — prot_lig_traj and ligand_traj updated in-place.")

    # ── 2. Ligand feature typing (RDKit) ─────────────────────────────────────

    def get_ligand_rings(self) -> List[Dict]:
        """Detect aromatic and aliphatic rings in the ligand using RDKit.

        Also populates self._ligand_aromatic_atom_names (atom names from the
        PDB monomer info) for use by get_atom_property().

        Returns
        -------
        list of dicts: [{'atoms': [idx, ...], 'atom_names': [str, ...], 'size': int, 'aromatic': bool}, ...]

        Note
        ----
        Atom indices are RDKit indices, not MDTraj indices.
        Index-name consistency to be validated during benchmarking.

        Manual override
        ---------------
        If auto-detection gives wrong results, assign a replacement list to
        ``self._manual_rings`` before calling this method (or before calling
        ``compute_aromatic_contacts``).  Each entry requires only
        ``'atom_names'`` and ``'aromatic'``; ``'size'`` is filled in
        automatically::

            sim._manual_rings = [
                {'atom_names': ['C1', 'C2', 'C3', 'C4', 'C5', 'C6'], 'aromatic': True},
                {'atom_names': ['C7', 'C8', 'N1', 'C9', 'C10'],       'aromatic': True},
            ]

        Set ``sim._manual_rings = None`` to revert to automatic detection.
        """
        if self._manual_rings is not None:
            aromatic_names = set()
            normalised = []
            for entry in self._manual_rings:
                names = list(entry['atom_names'])
                is_aromatic = bool(entry.get('aromatic', True))
                normalised.append({
                    'atoms':      list(entry.get('atoms', [])),
                    'atom_names': names,
                    'size':       len(names),
                    'aromatic':   is_aromatic,
                })
                if is_aromatic:
                    aromatic_names.update(names)
            self._ligand_aromatic_atom_names = aromatic_names
            print(f"get_ligand_rings: using {len(normalised)} manual ring(s) "
                  f"({sum(r['aromatic'] for r in normalised)} aromatic)")
            return normalised

        if self.ligand_mol is None:
            return []
        ring_info = self.ligand_mol.GetRingInfo()
        rings, aromatic_names = [], set()
        for ring_atoms in ring_info.AtomRings():
            ring_list   = list(ring_atoms)
            is_aromatic = all(
                self.ligand_mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring_list
            )
            atom_names = []
            for idx in ring_list:
                atom = self.ligand_mol.GetAtomWithIdx(idx)
                info = atom.GetMonomerInfo()
                name = info.GetName().strip() if info else atom.GetSymbol()
                atom_names.append(name)
            rings.append({
                'atoms': ring_list,
                'atom_names': atom_names,
                'size': len(ring_list),
                'aromatic': is_aromatic,
            })
            if is_aromatic:
                aromatic_names.update(atom_names)
        self._ligand_aromatic_atom_names = aromatic_names
        return rings

    def get_ligand_hbond_pairs(self) -> List[Tuple[int, int]]:
        """Detect H-bond donor pairs (heavy atom, H) in the ligand using RDKit.

        Returns
        -------
        list of (heavy_atom_idx, hydrogen_idx) tuples — RDKit indices.

        Note
        ----
        RDKit indices ≠ MDTraj indices — validate during benchmarking.
        """
        if self.ligand_mol is None:
            return []
        pairs = []
        for atom in self.ligand_mol.GetAtoms():
            if atom.GetSymbol() in ('N', 'O'):
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetSymbol() == 'H':
                        pairs.append((atom.GetIdx(), neighbor.GetIdx()))
        return pairs

    def get_atom_property(self, atom_name: str) -> str:
        """Classify a ligand atom by chemical property.

        Parameters
        ----------
        atom_name : str
            Atom name as it appears in the MDTraj topology.

        Returns
        -------
        'aromatic' | 'nonpolar' | 'polar'

        Note
        ----
        Call get_ligand_rings() before this method to populate the aromatic
        atom name set. Name matching between MDTraj and RDKit to be validated
        during benchmarking.
        """
        in_ring = atom_name in self._ligand_aromatic_atom_names
        if 'C' in atom_name:
            return 'aromatic' if in_ring else 'nonpolar'
        return 'aromatic' if in_ring else 'polar'

    # ── 3. Protein topology helpers ───────────────────────────────────────────

    def get_protein_rings(self) -> Tuple:
        """Get aromatic ring atom indices for all aromatic protein residues.

        Returns
        -------
        (protein_rings, protein_rings_index, ring_atoms_by_resname)
        - protein_rings        : list of atom index arrays (prot_lig context)
        - protein_rings_index  : list of residue indices
        - ring_atoms_by_resname: dict mapping residue name to MDTraj selection string
        """
        ring_atoms_by_resname = {
            'TYR': 'name CG CD1 CD2 CE1 CE2 CZ',
            'TRP': 'name CG CD1 NE1 CE2 CD2 CZ2 CE3 CZ3 CH2',
            'HIS': 'name CG ND1 CE1 NE2 CD2',
            'PHE': 'name CG CD1 CD2 CE1 CE2 CZ',
        }
        
        # Residues with aromatic rings - good for π-π interactions
        # TRP = Tryptophan (indole ring)
        # TYR = Tyrosine (phenol ring)
        # PHE = Phenylalanine (benzyl ring)
        # HIS = Histidine (imidazole ring)
        protein_rings, protein_rings_index = [], []
        aro_ca = self.prot_lig_top.select("resname TYR PHE HIS TRP and name CA")
        for i in aro_ca:
            atom = self.prot_lig_top.atom(i)
            sel  = ring_atoms_by_resname.get(atom.residue.name)
            if sel:
                ring = self.prot_lig_top.select(f"resid {atom.residue.index} and {sel}")
                protein_rings.append(ring)
                protein_rings_index.append(atom.residue.index)
        return protein_rings, protein_rings_index, ring_atoms_by_resname

    def get_protein_ligand_pairs(self) -> np.ndarray:
        """Return all (protein_residue_idx, ligand_residue_idx) index pairs."""
        return np.array(list(product(self.protein_residue_numbers, self.ligand_residue_numbers)))

    def get_residue_atoms_dict(self) -> Dict:
        """Map each protein residue label to its atom index array."""
        return {
            f'{r.name}_{r.resSeq + self.offset}': self.protein_top.select(f'resid {r.index}')
            for r in self.protein_top.residues
        }

    def get_hbonded_atoms_dict(self) -> Dict:
        """Map each heavy atom (bonded to H) to its hydrogen partner.

        Used when reconstructing H-bond pharmacophore features.
        """
        atom_withH = {}
        for bond in self.protein_traj.topology.bonds:
            if bond[1].element.symbol == 'H':
                atom_withH[bond[0]] = bond[1]
        return atom_withH

    def get_atom_name_dict(self) -> Dict:
        """Map atom index to '{name}_{resname}_{resSeq}' label (prot_lig context)."""
        return {
            atom.index: f'{atom.name}_{atom.residue.name}_{atom.residue.resSeq + self.offset}'
            for atom in self.prot_lig_top.atoms
        }

    def convert_ligand_atom_name(self, ligand_atom: str) -> np.ndarray:
        """Return MDTraj atom indices for a ligand atom name (prot_lig context)."""
        return self.prot_lig_top.select(f'resname {self.ligand_resname} and name {ligand_atom}')

    # ── 4. Contact probability ────────────────────────────────────────────────

    def compute_aromatic_contacts(self):
        """Compute aromatic contact probabilities between ligand and protein residues.

        Geometric criteria (cutoffs fixed per literature definition):
          p-stack : r ≤ 6.5 Å, θ ≤ 45°, φ ≤ 60°
          t-stack : r ≤ 7.5 Å, θ ≥ 75°, φ ≤ 60°
        Distances in nm (MDTraj convention); angles in degrees.

        Returns
        -------
        pd.DataFrame
            Indexed by residue name. Columns: aromatic_stacking,
            aromatic_stacking_error, aromatic_pstacking, aromatic_pstacking_error,
            aromatic_tstacking, aromatic_tstacking_error, plus per-ligand-ring
            columns '0', '0_error', '1', '1_error', ...
            Also stored on self.contact_probability.
        """
        # ── ligand aromatic rings → MDTraj atom indices ───────────────────────
        rings = self.get_ligand_rings()
        aromatic_rings = [r for r in rings if r['aromatic']]
        if not aromatic_rings:
            raise ValueError("No aromatic rings detected in ligand — "
                             "check that get_ligand_rings() returns aromatic=True rings")

        ligand_rings_conv = []
        for ring in aromatic_rings:
            indices = [int(self.convert_ligand_atom_name(name)[0])
                       for name in ring['atom_names']]
            ligand_rings_conv.append(indices)

        protein_rings, protein_rings_index, _ = self.get_protein_rings()

        n_lig    = len(aromatic_rings)
        n_frames = self.prot_lig_traj.n_frames

        # Cutoffs in nm
        p_stack_cutoff = 0.65   # 6.5 Å
        t_stack_cutoff = 0.75   # 7.5 Å

        # Ring-center distances are computed from raw xyz (no MDTraj PBC path).
        # This is correct only when the complex is already made whole (no atoms
        # split across periodic boundaries).  A PBC-corrected trajectory satisfies
        # this; verify your input if results look anomalous.

        # ── ring centres + normals: (n_frames, 2, 3) per ring ─────────────────
        lig_params = [
            get_ring_center_normal_trj_assign_atomid_new(
                self.prot_lig_traj.xyz[:, np.array(idx), :], 0, 1, 2)
            for idx in ligand_rings_conv
        ]
        prot_params = [
            get_ring_center_normal_trj_assign_atomid_new(
                self.prot_lig_traj.xyz[:, ring, :], 0, 1, 2)
            for ring in protein_rings
        ]

        # ── pairwise geometry: (n_frames, n_lig × n_prot) ────────────────────
        n_pairs   = n_lig * len(protein_rings)
        distances = np.zeros((n_frames, n_pairs))
        thetas    = np.zeros((n_frames, n_pairs))
        phis      = np.zeros((n_frames, n_pairs))

        for i, (lp, pp) in enumerate(itertools.product(lig_params, prot_params)):
            lig_c, prot_c = lp[:, 0, :], pp[:, 0, :]
            lig_n, prot_n = lp[:, 1, :], pp[:, 1, :]
            distances[:, i] = np.linalg.norm(lig_c - prot_c, axis=1)
            connect      = normvector_connect_new(prot_c, lig_c)
            theta        = np.rad2deg(angle_new(prot_n, lig_n))
            phi          = np.rad2deg(angle_new(prot_n, connect))
            thetas[:, i] = np.abs(theta) - 2 * (np.abs(theta) > 90.0) * (np.abs(theta) - 90.0)
            phis[:, i]   = np.abs(phi)   - 2 * (np.abs(phi)   > 90.0) * (np.abs(phi)   - 90.0)

        pstacked = np.zeros((n_frames, n_pairs), dtype=int)
        tstacked = np.zeros((n_frames, n_pairs), dtype=int)
        stacked  = np.zeros((n_frames, n_pairs), dtype=int)

        for j in range(n_pairs):
            r_p  = np.where(distances[:, j] <= p_stack_cutoff)[0]
            r_t  = np.where(distances[:, j] <= t_stack_cutoff)[0]
            e    = np.where(thetas[:, j] <= 45)[0]
            f    = np.where(phis[:, j]   <= 60)[0]
            g    = np.where(thetas[:, j] >= 75)[0]
            p_fr = np.intersect1d(np.intersect1d(e, f), r_p)
            t_fr = np.intersect1d(np.intersect1d(g, f), r_t)
            pstacked[p_fr, j] = 1
            tstacked[t_fr, j] = 1
            stacked[p_fr, j]  = 1
            stacked[t_fr, j]  = 1

        # ── split by ligand ring: each sub-array (n_frames, n_prot_rings) ─────
        stk_by_ring  = np.split(stacked,  n_lig, axis=1)
        pstk_by_ring = np.split(pstacked, n_lig, axis=1)
        tstk_by_ring = np.split(tstacked, n_lig, axis=1)

        # Fix overcounting: a residue contacted by multiple ligand rings counts once
        fix_stk  = (np.sum(np.stack(stk_by_ring,  axis=0), axis=0) != 0).astype(int)
        fix_pstk = (np.sum(np.stack(pstk_by_ring, axis=0), axis=0) != 0).astype(int)
        fix_tstk = (np.sum(np.stack(tstk_by_ring, axis=0), axis=0) != 0).astype(int)

        # ── map to full protein residue array (n_frames, n_protein_residues) ──
        n_res = self.protein_traj.n_residues
        stk_full  = np.zeros((n_frames, n_res), dtype=int)
        pstk_full = np.zeros((n_frames, n_res), dtype=int)
        tstk_full = np.zeros((n_frames, n_res), dtype=int)
        stk_by_ring_full = [np.zeros((n_frames, n_res), dtype=int) for _ in range(n_lig)]

        for ri, res_idx in enumerate(protein_rings_index):
            stk_full[:, res_idx]  = fix_stk[:, ri]
            pstk_full[:, res_idx] = fix_pstk[:, ri]
            tstk_full[:, res_idx] = fix_tstk[:, ri]
            for l in range(n_lig):
                stk_by_ring_full[l][:, res_idx] = stk_by_ring[l][:, ri]

        # stacking_contacts_dict used by compute_contact_probability() and define_pharmacophore()
        self.stacking_contacts_dict = {
            'sum': stk_full,
            **{str(l): stk_by_ring_full[l] for l in range(n_lig)},
        }

        self._stacked_full        = stk_full
        self._pstacked_full       = pstk_full
        self._tstacked_full       = tstk_full
        self._stacked_by_ring     = stk_by_ring_full
        self._aromatic_rings      = aromatic_rings
        self._protein_rings_index = protein_rings_index

        # ── aggregate to per-residue probabilities ────────────────────────────
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        stk_ave,  stk_be  = get_blockerrors_pyblock_nanskip(stk_full,  1.0)
        pstk_ave, pstk_be = get_blockerrors_pyblock_nanskip(pstk_full, 1.0)
        tstk_ave, tstk_be = get_blockerrors_pyblock_nanskip(tstk_full, 1.0)

        df = pd.DataFrame({
            'aromatic_stacking':        stk_ave,
            'aromatic_stacking_error':  stk_be,
            'aromatic_pstacking':       pstk_ave,
            'aromatic_pstacking_error': pstk_be,
            'aromatic_tstacking':       tstk_ave,
            'aromatic_tstacking_error': tstk_be,
        }, index=self.residue_names)

        for l in range(n_lig):
            ring_ave, ring_be = get_blockerrors_pyblock_nanskip(stk_by_ring_full[l], 1.0)
            df[str(l)]       = ring_ave
            df[f'{l}_error'] = ring_be

        self.aromatic_contact_probability = df
        return df

    def compute_contact_probability(self, cutoff: float = DEFAULT_CONTACT_CUTOFF,
                                    scheme: str = 'closest-heavy'):
        """Compute general distance-based protein–ligand contact probability.

        Uses MDTraj compute_contacts on all protein–ligand residue pairs.
        Contacts defined by distance < cutoff (nm).

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm. Default 0.6 nm (6 Å).
        scheme : str
            MDTraj contact scheme. Default 'closest-heavy'.

        Returns
        -------
        pd.DataFrame
            Indexed by residue name. Columns: contact_probability,
            contact_probability_error, Kd, Kd_error.
            Stored on self.contact_probability.

        Also stores
        -----------
        self.dual_contact_matrix : residue co-occurrence DataFrame
        self.kd                  : (KD_mM, KD_error_mM)
        self.bound_fraction      : (boundfrac, boundfrac_be)
        self.kd_over_time        : cumulative bound-fraction time series DataFrame
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        combined_pairs = self.get_protein_ligand_pairs()
        distances = np.asarray(
            md.compute_contacts(self.prot_lig_traj, combined_pairs, scheme=scheme)[0]
        ).astype(float)

        contact_matrix = (distances < cutoff).astype(int)
        del distances

        contact_probability_ave, contact_probability_be = \
            get_blockerrors_pyblock_nanskip(contact_matrix, 1.0)

        dual_contact = (contact_matrix.T @ contact_matrix) / len(contact_matrix)
        self.dual_contact_matrix = pd.DataFrame(
            dual_contact,
            index=self.residue_names,
            columns=self.residue_names,
        )

        contact_overTime = np.sum(contact_matrix, axis=1)
        kd, kd_over_time_df, bound_fraction = get_Kd(
            prot_lig_traj=self.prot_lig_traj,
            simulation_times=self.simulation_times,
            contact_overTime=contact_overTime,
        )

        self.kd             = kd
        self.kd_over_time   = kd_over_time_df
        self.bound_fraction = bound_fraction

        df = pd.DataFrame({
            'contact_probability':       contact_probability_ave,
            'contact_probability_error': contact_probability_be,
            'Kd':                        kd[0],
            'Kd_error':                  kd[1],
        }, index=self.residue_names)

        self.contact_probability = df
        return df

    def compute_all_atom_contacts(self, cutoff: float = DEFAULT_CONTACT_CUTOFF,
                                   save_pca: bool = False):
        """Compute all-heavy-atom protein–ligand contact probabilities.

        Computes pairwise distances between every ligand heavy atom and every
        protein heavy atom (no element filter — all non-H atoms included).
        Contacts defined by distance < cutoff (nm).

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm. Default 0.6 nm (6 Å).
        save_pca : bool
            When True, store a per-frame distance matrix averaged over ligand
            atoms as self.all_atom_distances_df — shape (n_frames, n_prot_heavy),
            columns are protein atom name labels, index is simulation time.
            Used as input to the PCA trajectory-subset algorithm.

        Returns
        -------
        pd.DataFrame
            Indexed by residue name. Columns: all_atom_contacts,
            all_atom_contacts_error.
            Also stored on self.all_atom_contact_probability.

        Also stores
        -----------
        self.all_atom_contact_frames          : (n_frames, n_lig_heavy * n_prot_heavy) binary
        self.all_atom_ligand_atom_probability : per-ligand-atom probability DataFrame
        self.all_atom_distances_df            : (n_frames, n_prot_heavy) mean distances (save_pca only)
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        prot_heavy_idx = self.prot_lig_top.select('protein and not element H')
        lig_heavy_idx  = self.all_ligand_atoms_noh

        n_lig_heavy  = len(lig_heavy_idx)
        n_prot_heavy = len(prot_heavy_idx)
        n_frames     = self.prot_lig_traj.n_frames

        print(f"All-atom contacts: {n_lig_heavy} ligand heavy × {n_prot_heavy} protein heavy "
              f"= {n_lig_heavy * n_prot_heavy:,} pairs")

        all_pairs = np.array(list(product(lig_heavy_idx, prot_heavy_idx)))
        all_contacts = np.asarray(
            md.compute_distances(self.prot_lig_traj, all_pairs)
        ).astype(np.float32)   # (n_frames, n_lig_heavy * n_prot_heavy)

        # PCA distance matrix: mean over ligand atoms → (n_frames, n_prot_heavy)
        # Created before binarizing so the raw float array is still available.
        all_atom_distances_df = None
        if save_pca:
            atom_names_dict = self.get_atom_name_dict()
            col_names = [atom_names_dict[a] for a in prot_heavy_idx]
            all_atom_distances_df = pd.DataFrame(
                all_contacts.reshape(n_frames, n_lig_heavy, n_prot_heavy).mean(axis=1),
                index=self.simulation_times,
                columns=col_names,
            )

        # Binary contacts and reshape
        all_contact_frames = np.where(all_contacts < cutoff, 1, 0)
        reshaped = all_contact_frames.reshape(n_frames, n_lig_heavy, n_prot_heavy)
        del all_contacts

        # Per-ligand-atom probability of any contact with any protein atom
        init_df = pd.DataFrame(
            index=[self.prot_lig_atom_names_dict.get(e, e) for e in self.all_ligand_atoms]
        )
        init_df['temp'] = 0
        init_df = init_df[~init_df.index.str.startswith('H')]
        lig_atom_prob_df = pd.DataFrame(
            index=[self.prot_lig_atom_names_dict.get(e, e) for e in lig_heavy_idx]
        )
        lig_atom_prob_df['Probability'] = (reshaped == 1).any(axis=-1).mean(axis=0)
        lig_atom_prob_df = pd.concat([init_df, lig_atom_prob_df], axis=1)
        lig_atom_prob_df.fillna(0, inplace=True)
        lig_atom_prob_df.drop('temp', axis=1, inplace=True)

        # OR-reduce to residue level: any ligand atom contacts this protein atom?
        any_lig_contact = reshaped.any(axis=1)   # (n_frames, n_prot_heavy)

        atom_to_res = np.array([
            self.prot_lig_top.atom(a).residue.index for a in prot_heavy_idx
        ])

        n_res = self.protein_traj.n_residues
        res_contacts = np.zeros((n_frames, n_res), dtype=np.int8)
        for j, res_idx in enumerate(atom_to_res):
            res_contacts[:, res_idx] |= any_lig_contact[:, j]

        ave, be = get_blockerrors_pyblock_nanskip(res_contacts.astype(float), 1.0)
        df = pd.DataFrame({'all_atom_contacts': ave}, index=self.residue_names)
        df['all_atom_contacts_error'] = be

        self.all_atom_contact_probability     = df
        self.all_atom_contact_frames          = all_contact_frames
        self.all_atom_ligand_atom_probability = lig_atom_prob_df
        self.all_atom_distances_df            = all_atom_distances_df

        return df

    def compute_hydrophobic_contacts(self, cutoff: float = DEFAULT_HYDROPHOBIC_CUTOFF,
                                     save_pca: bool = False):
        """Compute hydrophobic contact probabilities between ligand and protein residues.

        Atom-level distances between ligand C/S atoms and protein C atoms.
        Contacts defined by distance < cutoff (nm).

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm. Default 0.40 nm (4.0 Å).
        save_pca : bool
            When True, also compute per-frame per-residue mean distance DataFrame,
            stored on self.hydrophobic_distances_df (used for downstream PCA).

        Returns
        -------
        pd.DataFrame
            Indexed by residue name. Columns: hydrophobic_contacts,
            hydrophobic_contacts_error.
            Also stored on self.hydrophobic_contact_probability.
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        n_lig_hphob  = len(self.hydrophobic_atoms_ligand)
        n_prot_hphob = len(self.hydrophobic_atoms_protein)
        n_frames     = self.prot_lig_traj.n_frames

        hphob_contacts = np.asarray(
            md.compute_distances(self.prot_lig_traj, self.protein_ligand_hphob_pairs)
        ).astype(float)

        # Per-frame per-residue mean distance (optional — used for downstream PCA)
        hphob_distances_df = pd.DataFrame(
            index=self.simulation_times,
            columns=self.hydrophobic_residue_atoms_dict.keys(),
        )
        if save_pca:
            reshaped_dist = hphob_contacts.reshape(n_frames, n_lig_hphob, n_prot_hphob)
            mean_dist = np.mean(reshaped_dist, axis=1)  # (n_frames, n_prot_hphob)
            for res, atom_idxs in self.hydrophobic_residue_atoms_dict.items():
                hphob_distances_df[res] = np.mean(mean_dist[:, atom_idxs], axis=1)

        hphob_contact_frames = np.where(hphob_contacts < cutoff, 1, 0)
        reshaped_contact = hphob_contact_frames.reshape(n_frames, n_lig_hphob, n_prot_hphob)

        # Atom-level mean contact probability: (n_lig_hphob, n_prot_hphob)
        mean_atom_contacts = reshaped_contact.mean(0)
        atom_contact_df = pd.DataFrame(
            mean_atom_contacts,
            index=[self.prot_lig_atom_names_dict.get(e, e) for e in self.hydrophobic_atoms_ligand],
            columns=[self.prot_lig_atom_names_dict.get(e, e) for e in self.hydrophobic_atoms_protein],
        )

        # Per-ligand-atom probability of any hydrophobic contact with any protein atom
        init_df = pd.DataFrame(
            index=[self.prot_lig_atom_names_dict.get(e, e) for e in self.all_ligand_atoms]
        )
        init_df['temp'] = 0
        init_df = init_df[~init_df.index.str.startswith('H')]
        ligand_atom_prob_df = pd.DataFrame(
            index=[self.prot_lig_atom_names_dict.get(e, e) for e in self.hydrophobic_atoms_ligand]
        )
        ligand_atom_prob_df['Probability'] = (reshaped_contact == 1).any(-1).mean(0)
        ligand_atom_prob_df = pd.concat([init_df, ligand_atom_prob_df], axis=1)
        ligand_atom_prob_df.fillna(0, inplace=True)
        ligand_atom_prob_df.drop('temp', inplace=True, axis=1)

        # Map atom contacts to residues: (n_frames, n_protein_residues)
        # Reduce ligand dimension: is protein C atom j contacted by ANY lig C/S atom?
        any_lig_contact = reshaped_contact.any(axis=1)  # (n_frames, n_prot_hphob)

        # Residue index for each protein C atom — one lookup per atom, not per frame
        atom_to_res = np.array([
            self.prot_lig_top.atom(atom_idx).residue.index
            for atom_idx in self.hydrophobic_atoms_protein
        ])

        # OR-reduce into residue slots (n_frames, n_protein_residues)
        n_res = self.protein_traj.n_residues
        Hphob_res_contacts = np.zeros((n_frames, n_res), dtype=np.int8)
        for j, res_idx in enumerate(atom_to_res):
            Hphob_res_contacts[:, res_idx] |= any_lig_contact[:, j]

        hphob_ave, hphob_be = get_blockerrors_pyblock_nanskip(Hphob_res_contacts, 1.0)
        df = pd.DataFrame({'hydrophobic_contacts': hphob_ave}, index=self.residue_names)
        df['hydrophobic_contacts_error'] = hphob_be

        self.hydrophobic_contact_probability      = df
        self.hydrophobic_contact_frames           = hphob_contact_frames
        self.hydrophobic_atom_contact_probability = atom_contact_df
        self.hydrophobic_ligand_atom_probability  = ligand_atom_prob_df
        self.hydrophobic_distances_df             = hphob_distances_df

        return df

    def compute_hbond_contacts(self, ligand_hbond_donors=None,
                               distance_cutoff: float = 0.35,
                               angle_cutoff: float = 150,
                               save_pca: bool = False):
        """Compute H-bond contact probabilities between ligand and protein residues.

        Detects H-bonds per frame with Baker–Hubbard geometry criteria. Tracks:
          PD (protein donor) : protein N/O/S–H donates to ligand acceptor
          LD (ligand donor)  : ligand N/O/S–H donates to protein acceptor

        Parameters
        ----------
        ligand_hbond_donors : list of (int, int), optional
            Ligand H-bond donor pairs as (heavy_atom_RDKit_idx, H_RDKit_idx).
            Auto-detected from the ligand RDKit mol when not provided.
        distance_cutoff : float
            Donor–acceptor distance cutoff in nm. Default 0.35 nm (3.5 Å).
        angle_cutoff : float
            Donor–H–acceptor angle cutoff in degrees. Default 150°.
        save_pca : bool
            When True, store per-frame raw distances for downstream PCA in
            self.hbond_distances_df and self.hbond_residue_distances_df.

        Returns
        -------
        pd.DataFrame
            Indexed by residue name. Columns: Hbonds_average,
            Hbonds_average_error, Hbonds_PD_average, Hbonds_PD_average_error,
            Hbonds_LD_average, Hbonds_LD_average_error.
            Also stored on self.hbond_contact_probability.
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if _get_bond_triplets is None:
            raise ImportError(
                "mdtraj.geometry.hbond._get_bond_triplets not importable — "
                "ensure MDTraj supports the lig_donors= parameter"
            )
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        n_frames   = self.prot_lig_traj.n_frames
        n_residues = self.protein_traj.n_residues

        # Auto-detect ligand H-bond donors from RDKit mol when not supplied
        if ligand_hbond_donors is None:
            ligand_hbond_donors = self.get_ligand_hbond_pairs()

        # Convert RDKit indices → MDTraj prot_lig indices via _ligand_sel_idx
        # RDKit index i maps to _ligand_sel_idx[i] (same atom ordering)
        ligand_hbond_donors_conv = [
            [int(self._ligand_sel_idx[heavy]), int(self._ligand_sel_idx[h])]
            for heavy, h in ligand_hbond_donors
        ]

        # Topology-level donor/acceptor inventory (frame 0 only)
        hbond_donors_acceptors_dict = print_donors_acceptors(
            self.prot_lig_traj[0],
            angle_cutoff=angle_cutoff,
            lig_donor_index=ligand_hbond_donors_conv,
            sidechain_only=True,
            offset=self.offset,
        )

        # O(1) membership sets — exclude last 3 protein atoms (C-terminal cap)
        protein_sel_set = set(self._protein_sel_idx[:-3])
        ligand_sel_set  = set(self._ligand_sel_idx)

        HBond_PD    = np.zeros((n_frames, n_residues), dtype=np.int8)
        HBond_LD    = np.zeros((n_frames, n_residues), dtype=np.int8)
        Hbond_pairs_PD: dict = {}
        Hbond_pairs_LD: dict = {}
        raw_distances, bond_triplets_raw_ref = [], None

        for frame in tqdm(range(n_frames)):
            hbonds, frame_raw = baker_hubbard2(
                self.prot_lig_traj[frame],
                angle_cutoff=angle_cutoff,
                distance_cutoff=distance_cutoff,
                lig_donor_index=ligand_hbond_donors_conv,
            )

            if save_pca:
                if bond_triplets_raw_ref is None:
                    bond_triplets_raw_ref = frame_raw[0]
                raw_distances.append(frame_raw[1])

            for hbond in hbonds:
                # PD: protein donates H → ligand acceptor
                if hbond[0] in protein_sel_set and hbond[2] in ligand_sel_set:
                    donor   = self.prot_lig_top.atom(hbond[0])
                    acc     = self.prot_lig_top.atom(hbond[2])
                    res_idx = donor.residue.index
                    HBond_PD[frame, res_idx] = 1
                    add_hbond_pair(donor, acc, Hbond_pairs_PD, res_idx)

                # LD: ligand donates H → protein acceptor
                if hbond[0] in ligand_sel_set and hbond[2] in protein_sel_set:
                    donor   = self.prot_lig_top.atom(hbond[0])
                    acc     = self.prot_lig_top.atom(hbond[2])
                    res_idx = acc.residue.index
                    HBond_LD[frame, res_idx] = 1
                    add_hbond_pair(donor, acc, Hbond_pairs_LD, res_idx)

        HB_Total = HBond_PD + HBond_LD  # 0, 1, or 2 when both types co-occur

        # ── per-ligand-atom H-bond probability ───────────────────────────────
        onLigand_hbonds = pd.DataFrame(
            index=[self.prot_lig_atom_names_dict.get(e, e) for e in self.all_ligand_atoms]
        )
        onLigand_hbonds.index = onLigand_hbonds.index.str.split('_').str[0]
        onLigand_hbonds['LD'] = 0.0
        onLigand_hbonds['PD'] = 0.0

        for res_idx, donor_data in Hbond_pairs_LD.items():
            for lig_donor, prot_data in donor_data.items():
                key = str(lig_donor).split('-')[1]
                for _, frames in prot_data.items():
                    onLigand_hbonds.loc[key, 'LD'] += frames / n_frames

        for res_idx, donor_data in Hbond_pairs_PD.items():
            for prot_donor, lig_data in donor_data.items():
                for lig_acc, frames in lig_data.items():
                    key = str(lig_acc).split('-')[1]
                    onLigand_hbonds.loc[key, 'PD'] += frames / n_frames

        # ── PCA distances (optional) ──────────────────────────────────────────
        hbond_distances_df   = None
        residue_distances_df = None
        if save_pca and bond_triplets_raw_ref is not None:
            index_atom = [
                f'{self.prot_lig_top.atom(t[0]).residue.name}_'
                f'{self.prot_lig_top.atom(t[0]).residue.resSeq + self.offset}'
                f'|{self.prot_lig_top.atom(t[2]).residue.name}_'
                f'{self.prot_lig_top.atom(t[2]).residue.resSeq + self.offset}'
                for t in bond_triplets_raw_ref
            ]
            hbond_distances_df = pd.DataFrame(
                np.concatenate(raw_distances, axis=0).reshape(len(index_atom), -1),
                index=index_atom,
                columns=self.simulation_times,
            )
            new_pairs, dist_means = [], []
            ligand_name = self.ligand_resname
            for p0, p1 in tqdm(product(self.all_residue_names_dict.values(), repeat=2)):
                key = f'{p0}|{p1}'
                if key not in index_atom:
                    continue
                if p0 == p1:
                    hbond_distances_df.drop(key, inplace=True)
                    continue
                try:
                    n0, n1 = int(p0.split('_')[1]), int(p1.split('_')[1])
                except (IndexError, ValueError):
                    n0, n1 = 0, 0
                if abs(n0 - n1) <= 4 and ligand_name not in p0 and ligand_name not in p1:
                    hbond_distances_df.drop(key, inplace=True)
                    continue
                new_pairs.append(key)
                dist_means.append(hbond_distances_df.loc[key].mean())
                hbond_distances_df.drop(key, inplace=True)
            residue_distances_df = pd.DataFrame(dist_means, index=new_pairs)

        # ── aggregate to per-residue probabilities ────────────────────────────
        HBond_PD_ave, HBond_PD_be = get_blockerrors_pyblock_nanskip(HBond_PD.astype(float), 1.0)
        HBond_LD_ave, HBond_LD_be = get_blockerrors_pyblock_nanskip(HBond_LD.astype(float), 1.0)
        HBond_ave,    HBond_be    = get_blockerrors_pyblock_nanskip(HB_Total.astype(float), 1.0)

        df = pd.DataFrame({
            'Hbonds_average':          HBond_ave,
            'Hbonds_average_error':    HBond_be,
            'Hbonds_PD_average':       HBond_PD_ave,
            'Hbonds_PD_average_error': HBond_PD_be,
            'Hbonds_LD_average':       HBond_LD_ave,
            'Hbonds_LD_average_error': HBond_LD_be,
        }, index=self.residue_names)

        self.hbond_contact_probability     = df
        self.hbond_contact_frames_pd       = HBond_PD
        self.hbond_contact_frames_ld       = HBond_LD
        self.hbond_pairs_pd                = Hbond_pairs_PD
        self.hbond_pairs_ld                = Hbond_pairs_LD
        self.hbond_donors_acceptors        = hbond_donors_acceptors_dict
        self.hbond_ligand_atom_probability = onLigand_hbonds
        self.hbond_distances_df            = hbond_distances_df
        self.hbond_residue_distances_df    = residue_distances_df

        return df

    # ── 4c. DSSP secondary structure ──────────────────────────────────────────

    def compute_dssp(self, get_over_time: bool = False, weights=None):
        """Compute DSSP secondary structure probabilities per residue.

        Uses MDTraj compute_dssp (simplified scheme: H=helix, E=sheet, C=coil).
        Block errors computed with pyblock; falls back to zero error if unavailable.

        Parameters
        ----------
        get_over_time : bool
            When True, also compute per-frame mean helix/sheet fraction across all
            residues, stored as self.dssp_over_time_df (indexed by simulation time).
        weights : np.ndarray or str, optional
            Per-frame weights array or path to a text file. When provided, a
            'DSSP_helix_reweighted' column is added to the output DataFrame.

        Returns
        -------
        pd.DataFrame
            Indexed by residue name. Columns: DSSP_helix, DSSP_helix_error_up,
            DSSP_helix_error_low, DSSP_sheet, DSSP_sheet_error_up,
            DSSP_sheet_error_low (plus DSSP_helix_reweighted when weights given).
            Stored on self.dssp_df.

        Also stores
        -----------
        self.dssp_over_time_df : pd.DataFrame or None
            Indexed by simulation time (ps). Columns: DSSP_helix, DSSP_helix_error,
            DSSP_sheet, DSSP_sheet_error. Only populated when get_over_time=True.
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        print("Computing DSSP …")
        dssp = md.compute_dssp(self.protein_traj, simplified=True)
        print(f"  DSSP array: {dssp.shape}  (frames × residues)")

        helix_res, helix_time, helix_rew = dssp_convert(dssp, 'H', get_over_time, weights)
        sheet_res, sheet_time, _         = dssp_convert(dssp, 'E', get_over_time, weights)

        df = pd.DataFrame({
            'DSSP_helix':           helix_res[:, 0],
            'DSSP_helix_error_up':  helix_res[:, 0] + helix_res[:, 1],
            'DSSP_helix_error_low': helix_res[:, 0] - helix_res[:, 1],
            'DSSP_sheet':           sheet_res[:, 0],
            'DSSP_sheet_error_up':  sheet_res[:, 0] + sheet_res[:, 1],
            'DSSP_sheet_error_low': sheet_res[:, 0] - sheet_res[:, 1],
        }, index=self.residue_names)

        if helix_rew and len(helix_rew) == len(df):
            df['DSSP_helix_reweighted'] = helix_rew

        over_time_df = None
        if get_over_time and helix_time is not None and sheet_time is not None:
            over_time_df = pd.DataFrame({
                'DSSP_helix':       helix_time[:, 0],
                'DSSP_helix_error': helix_time[:, 1],
                'DSSP_sheet':       sheet_time[:, 0],
                'DSSP_sheet_error': sheet_time[:, 1],
            }, index=self.simulation_times)

        self.dssp_df           = df
        self.dssp_over_time_df = over_time_df
        return df

    # ── 4d. Gyration radius & Sα free energy surface ─────────────────────────

    def compute_gyration_salpha(self, helix_pdb, T: float = 300.0, nbins: int = 60,
                                rg_range: tuple = (0.9, 6.0),
                                salpha_range: tuple = (0.0, 60.0)):
        """Compute per-frame gyration radius + Sα and their 2D free energy surface.

        Gyration radius is computed on Cα atoms with equal weights.

        Sα measures local helical propensity via a 6-residue sliding-window RMSD
        switching function against a reference helix structure, summed over all
        windows:

            Sα_frame = Σ_i (1 − (rmsd_i/0.08)^8) / (1 − (rmsd_i/0.08)^12)

        where rmsd_i is the Cα RMSD of window i vs. the helix reference (nm) and
        0.08 nm is the switching threshold.

        The 2D FES is: ΔG = −k_B T ln P(Rg, Sα)  (kcal/mol), shifted so the
        minimum is zero.

        Parameters
        ----------
        helix_pdb : str or md.Trajectory
            Path to a PDB file (or an already-loaded MDTraj Trajectory) of the
            reference ideal helix. Must have the same number of residues as the
            protein trajectory.
        T : float
            Temperature in Kelvin for the ΔG calculation. Default 300 K.
        nbins : int
            Number of histogram bins along each axis. Default 30.
        rg_range : (float, float)
            Rg axis range in nm. Default (0.9, 6.0).
        salpha_range : (float, float)
            Sα axis range. Default (0.0, 60.0).

        Returns
        -------
        pd.DataFrame
            2D FES in kcal/mol. Index = Sα bin centres, columns = Rg bin
            centres (nm). Stored on self.gyration_salpha_fes_df.

        Also stores
        -----------
        self.rg_over_time     : np.ndarray, shape (n_frames,) — Rg in nm
        self.salpha_over_time : np.ndarray, shape (n_frames,) — total Sα per frame
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required — install: pip install pandas")

        # ── Gyration radius (Cα equal weights) ────────────────────────────
        print("Computing gyration radius (Cα) …")
        mass_ca = np.zeros(self.protein_traj.topology.n_atoms, dtype=np.float64)
        for i in self.protein_traj.topology.select("name CA"):
            mass_ca[i] = 1.0
        rg = md.compute_rg(self.protein_traj, masses=mass_ca)  # (n_frames,) nm
        print(f"  Rg: {rg.min():.3f} – {rg.max():.3f} nm  (mean {rg.mean():.3f})")

        # ── Sα: local helical propensity ───────────────────────────────────
        print("Computing Sα …")
        helix_traj = md.load(helix_pdb) if isinstance(helix_pdb, str) else helix_pdb

        ca_sel_prot  = self.protein_traj.topology.select("name CA")
        ca_sel_helix = helix_traj.topology.select("name CA")
        prot_ca      = self.protein_traj.atom_slice(ca_sel_prot)
        helix_ca     = helix_traj.atom_slice(ca_sel_helix)

        if prot_ca.n_atoms != helix_ca.n_atoms:
            raise ValueError(
                f"Protein has {prot_ca.n_atoms} Cα atoms but helix reference has "
                f"{helix_ca.n_atoms} — they must match."
            )

        n_ca = prot_ca.n_atoms
        window_rmsds = np.stack([
            md.rmsd(prot_ca, helix_ca,
                    atom_indices=helix_ca.topology.select(f"resid {i} to {i + 5}"))
            for i in tqdm(range(n_ca - 5), desc="Sα windows")
        ])  # (n_windows, n_frames) in nm

        x      = window_rmsds / 0.08
        sa     = (1.0 - x**8) / (1.0 - x**12)   # switching function per window
        salpha = sa.sum(axis=0)                   # (n_frames,) total Sα
        print(f"  Sα: {salpha.min():.2f} – {salpha.max():.2f}  (mean {salpha.mean():.2f})")

        # ── 2D free energy surface ────────────────────────────────────────
        print("Computing 2D FES (Rg vs Sα) …")
        counts, rg_edges, sa_edges = np.histogram2d(
            rg, salpha, bins=nbins,
            range=[rg_range, salpha_range],
            density=True,
        )
        kBT = 0.001987 * T
        fes  = -kBT * np.log(counts + 1e-6)
        fes -= fes.min()

        rg_centers = 0.5 * (rg_edges[:-1] + rg_edges[1:])
        sa_centers = 0.5 * (sa_edges[:-1] + sa_edges[1:])

        fes_df = pd.DataFrame(fes.T, index=sa_centers, columns=rg_centers)

        self.rg_over_time           = rg
        self.salpha_over_time       = salpha
        self.gyration_salpha_fes_df = fes_df
        return fes_df

    # ── 4e. Intra-protein contact map ─────────────────────────────────────────

    def compute_contact_map(self, cutoff: float = 1.2, scheme: str = 'closest-heavy',
                            distance_raw: bool = False):
        """Compute the intra-protein residue contact map.

        Parameters
        ----------
        cutoff : float
            Distance cutoff in nm. Residue pairs below this threshold count as
            in contact (ignored when distance_raw=True).
        scheme : str
            MDTraj contact scheme: 'closest-heavy' (default) or 'ca'.
        distance_raw : bool
            If True, store raw per-frame distances instead of the mean binary
            contact probability matrix.

        Stores
        ------
        contact_map_df     : (n_res × n_res) symmetric probability DataFrame
                             (populated when distance_raw=False)
        contact_map_raw_df : (n_frames × n_pairs) raw distance DataFrame
                             (populated when distance_raw=True)

        Returns
        -------
        pd.DataFrame
            contact_map_df or contact_map_raw_df depending on distance_raw.
        """
        import time as _time
        t0 = _time.time()

        n_res = self.protein_traj.n_residues
        indices = np.stack(np.triu_indices(n_res, 1), 1)

        if distance_raw:
            raw = md.compute_contacts(self.protein_traj, indices, scheme='closest-heavy')[0]
            pair_labels = [
                f"{self.residue_names_dict[i]}|{self.residue_names_dict[j]}"
                for i, j in indices
            ]
            self.contact_map_raw_df = pd.DataFrame(
                raw,
                index=self.simulation_times,
                columns=pair_labels,
            )
            self.contact_map_df = None
            print(f'\tContact map (raw distances) done in {round(_time.time() - t0, 2)} s')
            return self.contact_map_raw_df

        distances = md.compute_contacts(self.protein_traj, indices, scheme=scheme)[0]
        contacts  = (distances < cutoff).astype(np.float32)   # (n_frames, n_pairs)
        matrix    = np.zeros((self.protein_traj.n_frames, n_res, n_res), dtype=np.float32)
        matrix[:, indices[:, 0], indices[:, 1]] = contacts
        matrix   += matrix.transpose(0, 2, 1)
        mean_mat  = matrix.mean(axis=0)
        self.contact_map_df = pd.DataFrame(
            mean_mat,
            index=self.residue_names,
            columns=self.residue_names,
        )
        self.contact_map_raw_df = None
        print(f'\tContact map done in {round(_time.time() - t0, 2)} s')
        return self.contact_map_df

    # ── 4c. SASA ──────────────────────────────────────────────────────────────

    def compute_sasa(self, n_points: int = 960, probe_radius: float = 0.14):
        """Compute per-atom / per-residue protein SASA and per-atom ligand SASA.

        Uses the Shrake-Rupley rolling-probe algorithm.  Because the simulations
        are run without explicit solvent, SASA here is a *geometric occlusion*
        measure rather than a literal water-contact area:

        - **Protein SASA** (``protein_traj`` only): how much of each protein
          atom / residue surface is unobstructed by neighbouring protein atoms.
        - **Ligand SASA in protein context** (``prot_lig_traj``): the probe is
          blocked by both protein and ligand atoms, so the result answers "how
          buried is this ligand atom inside the protein pocket?"
          Low SASA → deeply embedded / well-protected.
          High SASA → pointing outward / solvent-exposed if water were present.

        Parameters
        ----------
        n_points : int
            Number of sphere points for Shrake-Rupley (higher = more accurate,
            slower). Default 960.
        probe_radius : float
            Probe sphere radius in nm (default 0.14 nm = 1.4 Å, water probe).

        Stores
        ------
        sasa_atoms_df        : DataFrame, shape (n_frames, n_protein_atoms), nm²
        sasa_residues_df     : DataFrame, shape (n_frames, n_protein_residues), nm²
        sasa_ligand_atoms_df : DataFrame, shape (n_frames, n_ligand_atoms), nm²
                               Columns are bare atom names (e.g. 'C1', 'N2') for
                               direct use in PyMOL selections.

        Returns
        -------
        (sasa_atoms_df, sasa_residues_df, sasa_ligand_atoms_df)
        """
        import time as _time
        t0 = _time.time()

        # ── Strip virtual sites (element 'VS') — not in MDTraj's radius table ─
        def _strip_vs(traj):
            real = [a.index for a in traj.topology.atoms
                    if a.element is not None and a.element.symbol != 'VS']
            if len(real) == traj.n_atoms:
                return traj, None          # nothing to strip
            return traj.atom_slice(real), real

        prot_traj_sasa, _    = _strip_vs(self.protein_traj)
        pl_traj_sasa,   pl_real = _strip_vs(self.prot_lig_traj) \
            if self.prot_lig_traj is not None else (None, None)

        # ── Protein: per-atom SASA ────────────────────────────────────────────
        sasa_atoms = md.shrake_rupley(
            prot_traj_sasa,
            probe_radius=probe_radius,
            n_sphere_points=n_points,
            mode='atom',
        )
        atom_labels = [
            f'sasa_atom_{a.index}_{a.name}_{a.residue.name}{a.residue.resSeq}'
            for a in prot_traj_sasa.topology.atoms
        ]
        self.sasa_atoms_df = pd.DataFrame(
            sasa_atoms,
            columns=atom_labels,
            index=self.simulation_times,
        )

        # ── Protein: per-residue SASA ─────────────────────────────────────────
        sasa_residues = md.shrake_rupley(
            prot_traj_sasa,
            probe_radius=probe_radius,
            n_sphere_points=n_points,
            mode='residue',
        )
        residue_labels = [
            f'sasa_residue_{r.index}_{r.name}{r.resSeq}'
            for r in prot_traj_sasa.topology.residues
        ]
        self.sasa_residues_df = pd.DataFrame(
            sasa_residues,
            columns=residue_labels,
            index=self.simulation_times,
        )

        # ── Ligand: per-atom SASA in protein context ──────────────────────────
        # Protein atoms occlude the probe → low SASA means buried in pocket.
        self.sasa_ligand_atoms_df = None
        if pl_traj_sasa is not None and self.all_ligand_atoms is not None:
            sasa_prot_lig = md.shrake_rupley(
                pl_traj_sasa,
                probe_radius=probe_radius,
                n_sphere_points=n_points,
                mode='atom',
            )
            # Remap ligand atom indices to the VS-stripped trajectory
            if pl_real is not None:
                orig2new = {orig: new for new, orig in enumerate(pl_real)}
                lig_indices_sasa = [orig2new[i] for i in self.all_ligand_atoms
                                    if i in orig2new]
            else:
                lig_indices_sasa = list(self.all_ligand_atoms)
            lig_sasa = sasa_prot_lig[:, lig_indices_sasa]
            lig_labels = [
                self.prot_lig_top.atom(idx).name
                for idx in self.all_ligand_atoms
                if pl_real is None or idx in set(pl_real)
            ]
            self.sasa_ligand_atoms_df = pd.DataFrame(
                lig_sasa,
                columns=lig_labels,
                index=self.simulation_times,
            )

        print(f'\tSASA done in {round(_time.time() - t0, 2)} s')
        return self.sasa_atoms_df, self.sasa_residues_df, self.sasa_ligand_atoms_df

    # ── 4b. PCA trajectory analysis ───────────────────────────────────────────

    def compute_pca_trajectory(self, pca_dim: int = 2, analysis_dim: int = 2,
                               n_clusters: int = 2, gaussian_kernel: bool = True,
                               gaussian_sigma: float = 0.3,
                               standard_scaler: bool = False):
        """Run PCA + K-means clustering on the all-atom contact distance matrix.

        Requires compute_all_atom_contacts(save_pca=True) to have been called first.

        A Gaussian kernel exp(−d²/2σ²) is applied to raw distances before PCA
        (recommended). An optional StandardScaler step can be enabled with
        ``standard_scaler=True`` to reproduce results from the original
        REST-Analysis codebase, where StandardScaler was applied inside
        compute_pca after the Gaussian kernel.

        Parameters
        ----------
        pca_dim : int
            Total number of principal components to compute. Default 2.
        analysis_dim : int
            Number of PC dimensions passed to K-means. Default 2.
        n_clusters : int
            Number of K-means clusters. Default 2.
        gaussian_kernel : bool
            Apply Gaussian kernel exp(−d²/2σ²) to distances before PCA.
            Default True.
        gaussian_sigma : float
            Kernel width in nm. Default 0.3 nm (3 Å).
        standard_scaler : bool
            Apply StandardScaler (zero mean, unit std per feature) after the
            Gaussian kernel and before PCA. Set to True to match the original
            REST-Analysis behaviour. Default False.

        Returns
        -------
        (projection, eigenvalues, eigenvectors)
            Also stored on self.pca_result and related attributes.
        """
        if self.all_atom_distances_df is None:
            raise RuntimeError(
                "Call compute_all_atom_contacts(save_pca=True) before compute_pca_trajectory()"
            )

        pca_matrix = self.all_atom_distances_df.to_numpy()
        print(f"PCA input: {pca_matrix.shape[0]} frames × {pca_matrix.shape[1]} features")

        if gaussian_kernel:
            print(f"Applying Gaussian kernel (σ = {gaussian_sigma} nm) …")
            pca_matrix = np.exp(-(pca_matrix ** 2) / (2 * gaussian_sigma ** 2))

        (projection, l, v), fes_df, fes_skl_df, dtraj, frames_cl, clustercenters, \
            clusters_fig, cluster_populations, silhouette = \
            compute_pca(pca_matrix, pca_dim=pca_dim, analysis_dim=analysis_dim,
                        n_clusters=n_clusters, standard_scaler=standard_scaler)

        self.pca_result              = (projection, l, v)
        self.pca_fes_df              = fes_df
        self.pca_fes_skl_df          = fes_skl_df
        self.pca_dtraj               = dtraj
        self.pca_frames_cl           = frames_cl
        self.pca_cluster_centers     = clustercenters
        self.pca_clusters_fig        = clusters_fig
        self.pca_cluster_populations = cluster_populations
        self.pca_silhouette_score    = silhouette

        print(f"  Silhouette score : {silhouette:.3f}")
        print(f"  Cluster sizes    : {[len(f) for f in frames_cl]}")

        return projection, l, v

    # ── 4f. Graph-based sub-clustering ────────────────────────────────────────
    # Second-level clustering on top of a PCA cluster subset: builds a
    # residue-contact-network graph per frame and clusters frames by Jaccard
    # distance between those networks. Run this on a PCA-sliced `sim_cl`
    # (see analysis.ipynb Section 6) to split a PCA cluster further, mirroring
    # the original REST-Analysis "PCA_C{i}_Graph{j}" sub-clustering scheme.

    def compute_graph_clustering(self, cutoff: float = 0.6, scheme: str = 'closest',
                                 use_ligand: bool = False,
                                 full_trajectory_frames: Optional[int] = None):
        """Build per-frame residue contact graphs and the Jaccard distance matrix.

        Step 1 of the graph-clustering pipeline (expensive — O(n_frames²)
        Jaccard computation). Run cluster_graph_clustering() afterwards to
        assign labels; re-running that second step with different clustering
        parameters does not require rebuilding the graphs/distance matrix.

        Parameters
        ----------
        cutoff : float
            Residue-residue closest-atom distance cutoff in nm defining a
            contact edge. Default 0.6 nm.
        scheme : str
            MDTraj contact scheme passed to md.compute_contacts. Default 'closest'.
        use_ligand : bool
            If True, build the contact network on self.prot_lig_traj (protein +
            ligand residues). If False (default), use self.protein_traj (protein only).
        full_trajectory_frames : int or None
            Frame count of the original, un-sliced simulation this trajectory was
            taken from (e.g. sim.protein_traj.n_frames, when this is called on a
            PCA-cluster subset sim_cl). Used by save_graph_cluster_trajectories()'s
            min_population_fraction filter. Defaults to this trajectory's own
            n_frames when None — i.e. assumes self is already the full simulation.

        Returns
        -------
        (D, graphs)
            D      : (n_frames, n_frames) Jaccard distance matrix
            graphs : list[nx.Graph], one per frame
            Also stored on self.graph_cluster_distance_matrix / self.graph_cluster_graphs.
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if not NETWORKX_AVAILABLE:
            raise ImportError("networkx required — install: pip install networkx")

        traj_attr = 'prot_lig_traj' if use_ligand else 'protein_traj'
        traj = getattr(self, traj_attr)
        if traj is None:
            raise RuntimeError("Call load() before compute_graph_clustering()")

        print(f"Building residue contact graphs ({'protein+ligand' if use_ligand else 'protein'}, "
              f"cutoff={cutoff} nm, scheme='{scheme}') …")

        n_res = traj.n_residues
        res_pairs = np.stack(np.triu_indices(n_res, 1), 1)

        print("Computing residue–residue closest-atom distances...")
        dists, used_pairs = md.compute_contacts(traj, contacts=res_pairs, scheme=scheme)
        used_pairs = [(int(i), int(j)) for i, j in used_pairs]

        print("Thresholding to contacts and building edge sets...")
        frame_edge_sets: List[Set[Tuple[int, int]]] = []
        for f in tqdm(range(traj.n_frames)):
            mask = dists[f] < cutoff
            edges = {used_pairs[k] for k, m in enumerate(mask) if m}
            frame_edge_sets.append(edges)

        graphs = _build_nx_graphs(traj.topology, frame_edge_sets)
        D = jaccard_distance_matrix(frame_edge_sets)

        self.graph_cluster_distance_matrix = D
        self.graph_cluster_graphs          = graphs
        self._graph_cluster_traj_attr      = traj_attr
        self._graph_cluster_full_n_frames  = (
            full_trajectory_frames if full_trajectory_frames is not None else traj.n_frames
        )

        return D, graphs

    def cluster_graph_clustering(self, clustering_method: str = 'agglomerative',
                                 n_clusters: Optional[int] = None, linkage: str = 'average',
                                 distance_threshold: float = 0.4, representative_by: str = 'max_degree',
                                 hdbscan_min_cluster_size: int = 10,
                                 save_gexf: bool = False, gexf_dir: str = 'graphs_gexf'):
        """Cluster frames from the distance matrix built by compute_graph_clustering().

        Step 2 of the graph-clustering pipeline (cheap — safe to re-run with
        different clustering parameters without recomputing D/graphs).

        Parameters
        ----------
        clustering_method : str
            'agglomerative' (default, scikit-learn) or 'hdbscan' (optional dependency).
        n_clusters : int or None
            Number of clusters for agglomerative clustering. Mutually exclusive
            with distance_threshold — set exactly one, leave the other as its default.
        linkage : str
            Agglomerative linkage criterion. Default 'average'.
        distance_threshold : float
            Agglomerative distance threshold, used only when n_clusters is None.
        representative_by : str
            'max_degree' (default) picks the highest mean-node-degree frame per
            cluster; any other value picks the frame with the most edges.
        hdbscan_min_cluster_size : int
            Only used when clustering_method='hdbscan'. Default 10.
        save_gexf : bool
            If True, write one .gexf file per frame graph to gexf_dir (viewable
            in Gephi/Cytoscape). Default False.

        Returns
        -------
        (graph_df, representatives, silhouette)
            graph_df        : pd.DataFrame — frame, time_ps, cluster, n_edges, mean_degree, Rg_nm
            representatives : dict — cluster_label -> representative frame index (local)
            silhouette      : float or None (None if <2 valid non-singleton clusters)
            Also stored on self.graph_cluster_labels / self.graph_cluster_df /
            self.graph_cluster_representatives / self.graph_cluster_silhouette.
        """
        if self.graph_cluster_distance_matrix is None or self.graph_cluster_graphs is None:
            raise RuntimeError(
                "Call compute_graph_clustering() before cluster_graph_clustering()"
            )

        D      = self.graph_cluster_distance_matrix
        graphs = self.graph_cluster_graphs
        traj   = getattr(self, self._graph_cluster_traj_attr)

        print(f"Clustering {len(graphs)} frames ({clustering_method}) …")
        labels = _cluster_by_distance(D, clustering_method, n_clusters, linkage,
                                      distance_threshold, hdbscan_min_cluster_size)
        silhouette, _ = silhouette_from_distance(D, labels)

        graph_df = pd.DataFrame({
            "frame":       np.arange(traj.n_frames),
            "time_ps":     traj.time,
            "cluster":     labels,
            "n_edges":     [g.number_of_edges() for g in graphs],
            "mean_degree": [np.mean([d for _, d in g.degree()]) if g.number_of_nodes() > 0 else 0.0
                            for g in graphs],
            "Rg_nm":       md.compute_rg(traj).flatten(),
        })

        representatives = _representative_frames(graphs, labels, criterion=representative_by)

        if save_gexf:
            _save_gexf(graphs, gexf_dir)
            print(f"Saved per-frame graphs to {gexf_dir}/*.gexf")

        self.graph_cluster_labels          = labels
        self.graph_cluster_df              = graph_df
        self.graph_cluster_representatives = representatives
        self.graph_cluster_silhouette      = silhouette

        print(f"  Silhouette score : {silhouette}")
        print(f"  Cluster sizes    :\n{graph_df['cluster'].value_counts().to_string()}")

        return graph_df, representatives, silhouette

    def save_graph_cluster_trajectories(self, output_dir: str,
                                        min_population_fraction: float = 0.01,
                                        clusters_to_save: Optional[int] = None):
        """Write per-cluster sub-trajectories from cluster_graph_clustering() results.

        For each cluster passing the min_population_fraction floor (all of
        them by default, optionally capped further to the top clusters_to_save
        by population), writes into {output_dir}/graph_clusters/:
          cluster_{rank}_trajectory.xtc  — full multi-frame subset
          cluster_{rank}_trajectory.gro  — single representative frame

        Parameters
        ----------
        min_population_fraction : float
            Drop clusters smaller than this fraction of the *original* full
            simulation (self._graph_cluster_full_n_frames, set by
            compute_graph_clustering()'s full_trajectory_frames argument —
            not just this trajectory's own frame count). Default 0.01 (1%).
        clusters_to_save : int or None
            If given, further cap to the top clusters_to_save by population
            among those passing the fraction floor. Default None (keep all
            that pass the floor).

        Clusters are ranked by population size, largest first (rank 0 =
        largest), matching the on-disk naming already used elsewhere in this
        project's output/ folders. The noise label (-1, HDBSCAN only) is
        never saved.

        Frame indices (subset_frames, representative_frame) are in the *local*
        index space of whichever trajectory compute_graph_clustering() used
        (self.protein_traj or self.prot_lig_traj) — already relative to a PCA
        cluster subset if this is called on a sim_cl slice, not a global index
        into some larger "complete" trajectory.

        Returns
        -------
        dict : cluster_label -> {trajectory_path, structure_path, representative_frame, subset_frames}
            Also stored on self.graph_cluster_saved_paths.
        """
        if self.graph_cluster_df is None or self.graph_cluster_representatives is None:
            raise RuntimeError(
                "Call cluster_graph_clustering() before save_graph_cluster_trajectories()"
            )

        traj = getattr(self, self._graph_cluster_traj_attr)
        out_dir = os.path.join(output_dir, 'graph_clusters')
        os.makedirs(out_dir, exist_ok=True)

        cluster_counts = self.graph_cluster_df['cluster'].value_counts().sort_values(ascending=False)
        cluster_counts = cluster_counts[cluster_counts.index != -1]

        min_frames = min_population_fraction * self._graph_cluster_full_n_frames
        clusters = cluster_counts[cluster_counts >= min_frames].index.tolist()
        dropped = cluster_counts[cluster_counts < min_frames]
        if len(dropped):
            print(f"  Dropping {len(dropped)} cluster(s) below {min_population_fraction*100:.1f}% "
                  f"of {self._graph_cluster_full_n_frames} frames ({min_frames:.0f} frames): "
                  f"{dropped.to_dict()}")

        if clusters_to_save is not None:
            clusters = clusters[:clusters_to_save]

        saved_paths = {}
        for rank, cluster in enumerate(clusters):
            subset_frames = self.graph_cluster_df.loc[
                self.graph_cluster_df['cluster'] == cluster, 'frame'
            ].tolist()
            representative_frame = self.graph_cluster_representatives[cluster]

            trajectory_path = os.path.join(out_dir, f'cluster_{rank}_trajectory.xtc')
            structure_path  = os.path.join(out_dir, f'cluster_{rank}_trajectory.gro')

            traj[subset_frames].save_xtc(trajectory_path)
            traj[representative_frame].save_gro(structure_path)

            print(f"  Cluster {cluster} → rank {rank}: {len(subset_frames)} frames, "
                  f"representative frame {representative_frame}")

            saved_paths[cluster] = {
                'trajectory_path':      trajectory_path,
                'structure_path':       structure_path,
                'representative_frame': representative_frame,
                'subset_frames':        subset_frames,
            }

        self.graph_cluster_saved_paths = saved_paths
        return saved_paths

    def compute_negative_space(self,
                               shell_inner_radius: float = 1.5,
                               shell_outer_radius: float = 5.0,
                               dx: float = 0.5,
                               protein_vdw_radius: float = 1.5,
                               min_free_fraction: float = 0.7):
        """Compute voxel-based protein occupancy and space classification around the ligand.

        Builds a 3D grid over the ligand + surrounding region, calculates the
        fraction of trajectory frames each voxel is occupied by protein, and
        classifies every shell voxel into one of four categories.

        GPU-accelerated when PyTorch + CUDA are available; falls back to NumPy.

        Parameters
        ----------
        shell_inner_radius : float
            Inner boundary of the analysis shell (Å from ligand surface). Default 1.5 Å.
        shell_outer_radius : float
            Outer boundary of the analysis shell (Å from ligand). Default 5.0 Å.
        dx : float
            Voxel edge length in Å. Default 0.5 Å.
        protein_vdw_radius : float
            Exclusion radius used to decide whether a protein atom occupies a voxel (Å).
            Default 1.5 Å.
        min_free_fraction : float
            Threshold separating growth space from contested space (0–1).
            Default 0.7: voxels free ≥ 70% of frames → growth space (protein occ < 30%).
            Set to 0.0 to treat the entire shell as growth space.

        Returns
        -------
        dict with keys:
          growth_space_mask    : (nx,ny,nz) bool  — protein occ < 30% (free ≥ min_free_fraction)
          contested_space_mask : (nx,ny,nz) bool  — protein occ ≥ 30% (free <  min_free_fraction)
          occupied_space_mask  : (nx,ny,nz) bool  — protein occ > 70%  (diagnostic, IDP-rarely-useful)
          full_shell_mask      : (nx,ny,nz) bool  — all shell voxels, no threshold
          free_fraction        : (nx,ny,nz) float — fraction of frames voxel is unoccupied
          protein_occupancy    : (nx,ny,nz) float — fraction of frames protein present
          min_dist_to_ligand   : (nx,ny,nz) float — Å to nearest ligand atom ever
          grid_info            : dict — xmin, xmax, nbins, dx, shell radii
          volumes              : dict — volumes in Å³ for each region
        Stored on self.negative_space_data.
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if self.prot_lig_traj is None:
            raise RuntimeError("Call load() before compute_negative_space()")

        trajectory    = self.prot_lig_traj
        ligand_atoms  = self._ligand_sel_idx
        protein_atoms = self._protein_sel_idx

        print(f"Shell: {shell_inner_radius}–{shell_outer_radius} Å | "
              f"voxel {dx} Å | excl. radius {protein_vdw_radius} Å | "
              f"free threshold {min_free_fraction*100:.0f}%")

        # ── Step 1: Grid setup ────────────────────────────────────────────────
        # Positions in Å (MDTraj stores nm)
        # Shape: (n_frames, n_atoms, 3)
        # Converts trajectory coordinates from nanometers to Angstroms (1 nm = 10 Å)
        ligand_pos  = trajectory.xyz[:, ligand_atoms,  :] * 10.0
        protein_pos = trajectory.xyz[:, protein_atoms, :] * 10.0

        # Define bounding box for grid (around ligand + shell)
        # Flatten all ligand atom positions across all frames into a 2D array
        # xyz coordinates of all ligand atoms across all frames
        # Shape: (n_frames * n_ligand_atoms, 3)
        # This combines all frames and atoms into one list of XYZ coordinates

            #  INPUT  ligand_pos  ── shape: (n_frames, n_ligand_atoms, 3)
        #
        #  ┌─ frame 0 ──────────────────────┐
        #  │  atom₀  │  x  │  y  │  z  │   │
        #  │  atom₁  │  x  │  y  │  z  │   │
        #  │   ...   │  .  │  .  │  .  │   │
        #  └─────────────────────────────── ┘
        #  ┌─ frame 1 ──────────────────────┐
        #  │  atom₀  │  x  │  y  │  z  │   │
        #  │  atom₁  │  x  │  y  │  z  │   │
        #  │   ...   │  .  │  .  │  .  │   │
        #  └────────────────────────────────┘
        #        ⋮
        #  ┌─ frame N ──────────────────────┐
        #  │  atom₀  │  x  │  y  │  z  │   │
        #  │   ...   │  .  │  .  │  .  │   │
        #  └────────────────────────────────┘
        #
        #                 │
        #       .reshape(-1, 3)       (-1 = n_frames × n_ligand_atoms)
        #                 │
        #                 ▼
        #
        #  OUTPUT  all_ligand_pos  ── shape: (n_frames × n_ligand_atoms, 3)
        #
        #  ┌─────────────────────────────────┐
        #  │  atom₀  │  x  │  y  │  z  │  ← frame 0
        #  │  atom₁  │  x  │  y  │  z  │
        #  │   ...   │  .  │  .  │  .  │
        #  ├─────────────────────────────────┤
        #  │  atom₀  │  x  │  y  │  z  │  ← frame 1
        #  │  atom₁  │  x  │  y  │  z  │
        #  │   ...   │  .  │  .  │  .  │
        #  ├─────────────────────────────────┤
        #  │   ...   │  .  │  .  │  .  │  ← frame N
        #  └─────────────────────────────────┘
        #
        #  → min/max over axis=0 gives global bounding box
        #    across ALL atoms and ALL frames in one step
        all_ligand_pos = ligand_pos.reshape(-1, 3)
               
        # Calculate grid extents - these define the 3D box containing everything
        # Find minimum X, Y, Z coordinates across all ligand atom positions
        # Then subtract the shell radii + 2Å buffer to create padding
        # Shape: (3,) - one value for X, one for Y, one for Z minimum
        # Result: e.g., [-15.2, -8.5, -22.3]

        #  all_ligand_pos  ── shape: (n_frames × n_ligand_atoms, 3)
        #
        #         col→     X        Y        Z
        #               ┌────────┬────────┬────────┐
        #  row 0        │ -12.1  │  -6.3  │ -19.8  │
        #  row 1        │  -8.4  │  -5.1  │ -21.0  │
        #  row 2        │ -10.7  │  -7.9  │ -18.5  │
        #   ⋮          │   ⋮    │   ⋮   │   ⋮   │
        #  row N        │  -9.2  │  -6.8  │ -20.1  │
        #               └────────┴────────┴────────┘
        #                    │         │        │
        #              min() ↓   min() ↓  min() ↓     .min(axis=0)  →  collapse rows
        #               ┌────────┬────────┬────────┐
        #               │ -12.1  │  -7.9  │ -21.0  │  shape: (3,)
        #               └────────┴────────┴────────┘
        #                    │         │        │
        #          - 5.0 Å   ↓  - 5.0  ↓  - 5.0 ↓    shell_outer_radius = 5.0
        #          - 2.0 Å   ↓  - 2.0  ↓  - 2.0 ↓    buffer
        #               ┌────────┬────────┬────────┐
        #         xmin  │ -19.1  │ -14.9  │ -28.0  │  shape: (3,)
        #               └────────┴────────┴────────┘
        #
        #  → defines the lower-left-front corner of the 3D voxel grid

            #   Y
        #   ▲
        #   │                                                          
        #   │   xmin                                         xmax     
        #   │    ├──────────────────────────────────────────────┤     
        #   │    │◄─ 2Å ─►◄───── shell_outer_radius ─────►│    │     
        #   │    │         │                               │    │     
        #   │    │         │    · · ligand atoms · ·       │    │     
        #   │    │         │  ·   ╔═══════════╗   ·        │    │     
        #   │    │         │ ·    ║  ·  ·  ·  ║    ·       │    │     
        #   │    │         │ ·    ║ · ligand· ║    ·       │    │     
        #   │    │         │ ·    ║  ·  ·  ·  ║    ·       │    │     
        #   │    │         │  ·   ╚═══════════╝   ·        │    │     
        #   │    │         │    · · · · · · · ·             │    │     
        #   │    │         │                               │    │     
        #   │    │         │◄── growth space shell ────────►│    │
        #   │    │◄────────── voxel grid bounding box ─────────►│     
        #   │    │                                              │     
        #   └────┴──────────────────────────────────────────────────► X
        #
        #   ╔═══╗  ligand heavy atoms (min/max define raw extent)
        #   · · ·  shell region  (shell_outer_radius from any ligand atom)
        #   │   │  2 Å safety buffer  (ensures no voxel falls outside grid)
        #
        #   xmin[X] = min(all atom X coords)  - shell_outer_radius  - 2.0
        #   xmin[Y] = min(all atom Y coords)  - shell_outer_radius  - 2.0
        #   xmin[Z] = min(all atom Z coords)  - shell_outer_radius  - 2.0
        xmin = all_ligand_pos.min(axis=0) - shell_outer_radius - 2.0
        
        # Find maximum X, Y, Z coordinates across all ligand atom positions  
        # Then add the shell radii + 2Å buffer to create padding
        # Shape: (3,) - one value for X, one for Y, one for Z maximum
        # Result: e.g., [18.7, 12.3, 25.1]
        xmax = all_ligand_pos.max(axis=0) + shell_outer_radius + 2.0
        
        # Determine number of voxels in each dimension
        # nbins = ceil((xmax - xmin) / dx)
        # Example: if X-range = 34 Å and dx = 0.5 Å, nbins[0] = 68
        nbins = np.ceil((xmax - xmin) / dx).astype(int)

        print(f"Grid: {nbins}  ({np.prod(nbins):,} voxels)")

        # Create coordinate arrays for each dimension
        # x, y, z are 1D arrays of voxel center coordinates
        # Example for X: [-15.2, -14.7, -14.2, ..., 18.2, 18.7]
        x, y, z = (np.arange(nbins[i]) * dx + xmin[i] for i in range(3))
        
        # Create 3D meshgrid of all voxel positions
        # xx, yy, zz each have shape (nbins[0], nbins[1], nbins[2])
        # Each element is the coordinate value at that grid position
        xx, yy, zz    = np.meshgrid(x, y, z, indexing='ij')
        
        # Stack the three coordinate grids into one 4D array
        # voxel_coords has shape (nbins[0], nbins[1], nbins[2], 3)
        # Each element is [x_coord, y_coord, z_coord] for that voxel center
        voxel_coords  = np.stack([xx, yy, zz], axis=-1)
        
        # Reshape voxel grid for distance calculation
        # From 4D array (nbins[0], nbins[1], nbins[2], 3)
        # To 2D array (n_voxels, 3) where n_voxels = product of nbins
        voxel_flat    = voxel_coords.reshape(-1, 3)

        # ── Step 2: Minimum ligand distance per voxel ────────────────────────
        print("Step 1/3: ligand distance map …")

        # Initialize distance array with infinity
        # Will be progressively updated with actual distances
        # Shape: (nbins[0], nbins[1], nbins[2])
        #  min_dist_to_ligand  ── shape: (nbins[0], nbins[1], nbins[2])
        #
        #              Z
        #             ╱
        #            ╱
        #    ┌──────┬──────┬──────┐
        #   ╱  ∞   ╱  ∞   ╱  ∞  ╱│
        #  ┌──────┬──────┬──────┐ │  Y
        #  │  ∞   │  ∞   │  ∞   │ ┤ ──►
        #  │      │      │      │╱│
        #  ├──────┼──────┼──────┤ │
        #  │  ∞   │  ∞   │  ∞   │ ┤
        #  │      │      │      │╱│
        #  ├──────┼──────┼──────┤ │
        #  │  ∞   │  ∞   │  ∞   │ ┘
        #  └──────┴──────┴──────┘
        #  │
        #  ▼ X
        #
        #  Every voxel starts at ∞.
        #  As we loop over frames, each voxel is updated with the distance
        #  to the nearest ligand atom found so far  →  running minimum:
        #
        #  frame 0 ──► min_dist = min( ∞,   d₀ )  =  d₀
        #  frame 1 ──► min_dist = min( d₀,  d₁ )  =  d₀  (if d₀ < d₁)
        #  frame 2 ──► min_dist = min( d₀,  d₂ )  =  d₂  (if d₂ < d₀)
        #    ⋮
        #  frame N ──► min_dist = closest the ligand ever came to this voxel
        min_dist_to_ligand = np.full(nbins, np.inf, dtype=np.float32)

        if USE_GPU:
            # ---- GPU-ACCELERATED PATH (PyTorch + CUDA) ----
            # Move voxel grid to GPU once (stays resident for all frames)
            voxel_flat_gpu = torch.tensor(voxel_flat, dtype=torch.float32, device=GPU_DEVICE)
            min_dist_gpu   = torch.full((len(voxel_flat),), float('inf'),
                                        dtype=torch.float32, device=GPU_DEVICE)
            
            # With ~50 ligand atoms, the full distance matrix fits in GPU memory
            # (1M voxels × 50 atoms × 4 bytes = ~200 MB)
            gpu_chunk = 500_000

            #  trajectory.xyz  ── shape: (n_frames, n_atoms, 3)  [CPU, NumPy]
            #
            #  ┌─ frame 0 ─┐  ┌─ frame 1 ─┐       ┌─ frame N ─┐
            #  │ all atoms │  │ all atoms │  · · · │ all atoms │
            #  └───────────┘  └───────────┘       └───────────┘
            #        │               │                   │
            #        │  tqdm loop ───┴───────────────────┘
            #        │  frame_idx = 0, 1, 2, ... N
            #        ▼
            #
            #  ligand_pos[frame_idx]  ── shape: (n_ligand_atoms, 3)  [CPU, NumPy]
            #
            #  ┌─────────┬──────┬──────┬──────┐
            #  │  atom₀  │  x   │  y   │  z   │
            #  │  atom₁  │  x   │  y   │  z   │
            #  │   ...   │  .   │  .   │  .   │
            #  │  atomₙ  │  x   │  y   │  z   │
            #  └─────────┴──────┴──────┴──────┘
            #        │
            #        │  torch.tensor(..., device=GPU_DEVICE)
            #        │  ← copies this slice from RAM to VRAM each frame
            #        ▼
            #
            #  frame_ligand_gpu  ── shape: (n_ligand_atoms, 3)  [GPU, torch.float32]
            #
            #  ┌─────────┬──────┬──────┬──────┐
            #  │  atom₀  │  x   │  y   │  z   │  ┐
            #  │  atom₁  │  x   │  y   │  z   │  │ VRAM
            #  │   ...   │  .   │  .   │  .   │  │
            #  │  atomₙ  │  x   │  y   │  z   │  ┘
            #  └─────────┴──────┴──────┴──────┘
            #
            #  NOTE: only one frame is resident on GPU at a time → low VRAM footprint.
            #        the voxel grid (voxel_flat_gpu) stays loaded across all frames.
            #
            for fi in tqdm(range(trajectory.n_frames)):
                frame_lig = torch.tensor(ligand_pos[fi], dtype=torch.float32, device=GPU_DEVICE)
                
                #  voxel_flat_gpu  ── shape: (n_voxels, 3)  [GPU]
                #
                #  ┌────────────────────────────────────────────────┐
                #  │ voxel₀  │ voxel₁  │ · · · │ voxel_M │ · · ·  │
                #  └────────────────────────────────────────────────┘
                #   └────────────────┘ └───────────────────────────┘
                #      chunk i=0              chunk i=1  · · ·
                #      size: gpu_chunk_size
                #           │
                #           ▼
                #  chunk  ── shape: (gpu_chunk_size, 3)
                #
                #  ┌──────┬───┬───┬───┐       frame_ligand_gpu  ── shape: (n_ligand_atoms, 3)
                #  │  v₀  │ x │ y │ z │
                #  │  v₁  │ x │ y │ z │       ┌──────┬───┬───┬───┐
                #  │  v₂  │ x │ y │ z │       │  a₀  │ x │ y │ z │
                #  │  ... │ . │ . │ . │       │  a₁  │ x │ y │ z │
                #  │  vₙ  │ x │ y │ z │       │  ... │ . │ . │ . │
                #  └──────┴───┴───┴───┘       └──────┴───┴───┴───┘
                #           │                          │
                #           └──────── cdist ───────────┘
                #                        │
                #                        ▼
                #  dists  ── shape: (gpu_chunk_size, n_ligand_atoms)
                #
                #            a₀      a₁      a₂    · · ·   aₙ
                #  v₀  ┌  [ 3.2  │  7.1  │  4.5  │ · · · │ 2.1 ]
                #  v₁  │  [ 5.8  │  2.3  │  8.0  │ · · · │ 6.4 ]
                #  v₂  │  [ 1.9  │  4.7  │  3.3  │ · · · │ 5.2 ]
                #  ... │  [  .   │   .   │   .   │       │  .  ]
                #  vₙ  └  [ 6.1  │  3.9  │  2.7  │ · · · │ 4.8 ]
                #                        │
                #              .min(dim=1).values   ← minimum over atoms (columns)
                #                        │
                #                        ▼
                #  min_dist_chunk  ── shape: (gpu_chunk_size,)
                #
                #  [ 2.1,  2.3,  1.9, · · · ]   ← closest ligand atom per voxel
                #                        │
                #              torch.minimum(current, new)   ← running min across frames
                #                        │
                #                        ▼
                #  min_dist_gpu[i:i+gpu_chunk_size]  updated in-place

                #  ════════════════════════════════════════════════════════════════
                #  GOAL: for every voxel in the grid, find the closest distance
                #        that ANY ligand atom ever reached across ALL frames.
                #  ════════════════════════════════════════════════════════════════
                #
                #  INPUT
                #  ─────
                #  voxel_flat_gpu  ── (n_voxels, 3)      fixed grid of 3D points [GPU]
                #  ligand_pos      ── (n_frames, n_ligand_atoms, 3)  trajectory   [CPU]
                #
                #  PROCESS  (two nested loops)
                #  ───────
                #  outer loop → one frame at a time  (upload ligand slice to GPU)
                #  inner loop → one chunk of voxels at a time  (fit in VRAM)
                #
                #  for each (frame, chunk):
                #    cdist  →  full (chunk_size × n_ligand_atoms) distance matrix
                #    min(dim=1)  →  closest atom per voxel  for this frame
                #    torch.minimum  →  update running minimum across all frames
                #
                #  OUTPUT
                #  ──────
                #  min_dist_gpu  ── (n_voxels,)  [GPU]
                #
                #  each element = the shortest distance ever recorded between
                #  that voxel and any ligand atom, over the entire trajectory.
                #
                #  Example:
                #  ┌──────────┬───────────────────────────────────────────────┐
                #  │ voxel 0  │  0.3 Å  → inside ligand body                 │
                #  │ voxel 1  │  1.8 Å  → just outside ligand surface        │
                #  │ voxel 2  │  3.4 Å  → in shell region  ← growth candidate│
                #  │ voxel 3  │  8.1 Å  → far from ligand, outside shell     │
                #  └──────────┴───────────────────────────────────────────────┘
                #
                #  This map is then thresholded in Step 3 to define the shell:
                #  shell_inner_radius ≤ min_dist ≤ shell_outer_radius
                # Transfer result back to CPU
                for i in range(0, len(voxel_flat_gpu), gpu_chunk):
                    chunk = voxel_flat_gpu[i:i+gpu_chunk]
                    # torch.cdist uses optimised GEMM-based algorithm on GPU
                    # Result shape: (chunk_size, n_ligand_atoms)
                    dists = torch.cdist(chunk, frame_lig)
                    min_dist_gpu[i:i+gpu_chunk] = torch.minimum(
                        min_dist_gpu[i:i+gpu_chunk], dists.min(dim=1).values)
            min_dist_to_ligand = min_dist_gpu.cpu().numpy().reshape(nbins)
            del voxel_flat_gpu, min_dist_gpu, frame_lig
            torch.cuda.empty_cache()
        else:
            chunk = 10_000
            for fi in tqdm(range(trajectory.n_frames)):
                fp = ligand_pos[fi]
                for i in range(0, len(voxel_flat), chunk):
                    ch = voxel_flat[i:i+chunk]
                    d  = np.linalg.norm(
                        ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2
                    ).min(axis=1)
                    sl = min_dist_to_ligand.ravel()[i:i+chunk]
                    min_dist_to_ligand.ravel()[i:i+chunk] = np.minimum(sl, d)

        # ── Step 3: Shell mask ────────────────────────────────────────────────
            
        # Define shell: voxels within the shell distance from ligand
        # in_shell = True only for voxels between inner & outer radius
        # Shape: (nbins[0], nbins[1], nbins[2]) of Boolean values
        # Example:
        #   - distance = 0.5 Å → in_shell = False (inside ligand)
        #   - distance = 2.0 Å → in_shell = True (in shell)
        #   - distance = 6.0 Å → in_shell = False (outside shell)
            #  CROSS-SECTION VIEW (any 2D slice through the ligand centre)
        #
        #  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        #  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        #  ░░░░░░░░░░   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ░░░░░░░░░░░░░░░░
        #  ░░░░░░░  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ░░░░░░░░░░░░░
        #  ░░░░░  ▓▓▓▓▓▓▓▓▓   ┌─────────┐   ▓▓▓▓▓▓▓  ░░░░░░░░░░░
        #  ░░░░  ▓▓▓▓▓▓▓▓   ╔═══════════╗   ▓▓▓▓▓▓▓▓  ░░░░░░░░░░
        #  ░░░░ ▓▓▓▓▓▓▓▓  ╔═╝           ╚═╗  ▓▓▓▓▓▓▓▓ ░░░░░░░░░░
        #  ░░░ ▓▓▓▓▓▓▓▓  ║   · ligand ·  ║  ▓▓▓▓▓▓▓▓ ░░░░░░░░░░
        #  ░░░ ▓▓▓▓▓▓▓▓  ║   · atoms  ·  ║  ▓▓▓▓▓▓▓▓ ░░░░░░░░░░
        #  ░░░ ▓▓▓▓▓▓▓▓  ╚═╗           ╔═╝  ▓▓▓▓▓▓▓▓ ░░░░░░░░░░
        #  ░░░░ ▓▓▓▓▓▓▓▓  ╚═══════════╝   ▓▓▓▓▓▓▓▓ ░░░░░░░░░░
        #  ░░░░  ▓▓▓▓▓▓▓▓   └─────────┘   ▓▓▓▓▓▓▓▓  ░░░░░░░░░░
        #  ░░░░░  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ░░░░░░░░░░░
        #  ░░░░░░░  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓  ░░░░░░░░░░░░░
        #  ░░░░░░░░░░░   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓   ░░░░░░░░░░░░░░░░
        #  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░
        #
        #  ╔═══╗  ligand body         dist < shell_inner_radius   in_shell = False
        #  ▓▓▓▓▓  shell region        shell_inner ≤ dist ≤ shell_outer  in_shell = True  ← growth candidates
        #  ░░░░░  outside shell       dist > shell_outer_radius   in_shell = False
        #
        #  ◄────────────────────────────────────────────────────────►
        #        │◄── inner (1.5Å) ──►│◄──── shell (3.5Å) ────►│
        #        0                   1.5Å                      5.0Å
        #
        in_shell = ((min_dist_to_ligand >= shell_inner_radius) &
                    (min_dist_to_ligand <= shell_outer_radius))
        print(f"Shell voxels: {in_shell.sum():,}")

        # ── Step 4: Protein occupancy within shell ────────────────────────────
        print("Step 2/3: protein occupancy …")
        
        # For each voxel in shell, count how many frames protein occupies it
        # Shape: (nbins[0], nbins[1], nbins[2]) of integer counts (0 to n_frames)
        protein_occupancy = np.zeros(nbins, dtype=np.float32)
        
        # Only check shell voxels (much faster - skip interior & exterior)
        #  voxel_coords  ── shape: (nbins[0], nbins[1], nbins[2], 3)   [ALL voxels]
        #  in_shell      ── shape: (nbins[0], nbins[1], nbins[2])      [Bool mask]
        #
        #   in_shell (2D slice shown):
        #
        #   ┌─────────────────────────────────────┐
        #   │  F  F  F  F  F  F  F  F  F  F   F  │
        #   │  F  F  F  T  T  T  T  T  F  F   F  │
        #   │  F  F  T  T  F  F  F  T  T  F   F  │
        #   │  F  T  T  F  F  F  F  F  T  T   F  │
        #   │  F  T  F  F  F  F  F  F  F  T   F  │  F = outside shell (skip)
        #   │  F  T  F  F  F ╔══╗ F  F  F  T  F  │  T = in shell     (keep)
        #   │  F  T  F  F  F ║  ║ F  F  F  T  F  │
        #   │  F  T  T  F  F ╚══╝ F  F  T  T  F  │
        #   │  F  F  T  T  F  F  F  T  T  F   F  │
        #   │  F  F  F  T  T  T  T  T  F  F   F  │
        #   │  F  F  F  F  F  F  F  F  F  F   F  │
        #   └─────────────────────────────────────┘
        #                    │
        #          boolean indexing / np.where
        #                    │
        #          ┌─────────┴──────────────────────┐
        #          ▼                                ▼
        #
        #  shell_coords                     shell_indices
        #  shape: (n_shell_voxels, 3)       shape: (n_shell_voxels,)
        #
        #  ┌──────┬───┬───┬───┐             ┌──────────────────────┐
        #  │  v₀  │ x │ y │ z │             │  42                  │  ← flat index in
        #  │  v₁  │ x │ y │ z │             │  43                  │    ravel()ed grid
        #  │  v₂  │ x │ y │ z │             │  57                  │
        #  │  ... │ . │ . │ . │             │  ...                 │
        #  │  vₙ  │ x │ y │ z │             │  N                   │
        #  └──────┴───┴───┴───┘             └──────────────────────┘
        #  3D coordinates of                position in the flat 1D
        #  each shell voxel                 array  →  used later to
        #  → input to cdist                 scatter results back into
        #                                   protein_occupancy grid
        #
        #  WHY: avoids computing protein distances for interior voxels
        #       (always blocked) and exterior voxels (always free) —
        #       only the shell ~10-20% of the grid actually matters.
        shell_coords  = voxel_coords[in_shell]
        shell_indices = np.where(in_shell.ravel())[0]

        if USE_GPU:
            shell_coords_gpu  = torch.tensor(shell_coords,  dtype=torch.float32, device=GPU_DEVICE)
            shell_indices_gpu = torch.tensor(shell_indices, dtype=torch.long,    device=GPU_DEVICE)
            occupancy_gpu     = torch.zeros(int(np.prod(nbins)), dtype=torch.float32, device=GPU_DEVICE)
            # Allocates a 1D accumulator array on GPU, length = total number of voxels.
            # Flat (1D) rather than 3D to allow direct indexed assignment later.
            # Initialized to zeros.

            # Chunk size depends on n_protein_atoms to avoid GPU OOM
            # Budget ~4 GB for the distance matrix: chunk × n_protein × 4 bytes
            n_prot    = len(protein_atoms)
            gpu_chunk = max(1_000, min(200_000, int(4e9 / (n_prot * 4))))
            for fi in tqdm(range(trajectory.n_frames)):
                # Uploads the protein atom coordinates for the current frame to GPU. Shape: (n_protein_atoms, 3)
                frame_prot = torch.tensor(protein_pos[fi], dtype=torch.float32, device=GPU_DEVICE)
                
                #  ════════════════════════════════════════════════════════════════
                #  GOAL: for every shell voxel, count in how many frames at least
                #        one protein atom is within protein_vdw_radius (1.5 Å).
                #  ════════════════════════════════════════════════════════════════
                #
                #  shell_coords_gpu  ── (n_shell_voxels, 3)  [GPU]   coordinates
                #  shell_indices_gpu ── (n_shell_voxels,)    [GPU]   flat positions in full grid
                #
                #  ┌──────────────────────────────────────────────────────────┐
                #  │ v₀ │ v₁ │ v₂ │ · · · │ vₘ │ vₘ₊₁ │ · · · │ vₙ │ · · · │  shell_coords_gpu
                #  └──────────────────────────────────────────────────────────┘
                #   └──────────────────┘ └────────────────────┘
                #        chunk i=0              chunk i=1  · · ·
                #              │
                #              ▼
                #  chunk  ── (gpu_chunk_size, 3)        idx ── (gpu_chunk_size,)  flat indices
                #
                #  ┌──────┬───┬───┬───┐                ┌────────────────────┐
                #  │  v₀  │ x │ y │ z │                │  42                │
                #  │  v₁  │ x │ y │ z │                │  43                │
                #  │  ... │ . │ . │ . │                │  ...               │
                #  └──────┴───┴───┴───┘                └────────────────────┘
                #              │
                #              │   torch.cdist(chunk, frame_protein_gpu)
                #              ▼
                #
                #  dists  ── shape: (gpu_chunk_size, n_protein_atoms)
                #
                #            p₀      p₁      p₂    · · ·   pₙ
                #  v₀  ┌  [ 0.9  │  4.2  │  6.1  │ · · · │ 3.3 ]
                #  v₁  │  [ 3.1  │  1.1  │  2.8  │ · · · │ 5.7 ]   each cell = distance
                #  v₂  │  [ 5.4  │  6.0  │  0.7  │ · · · │ 4.1 ]   voxel → protein atom
                #  ... │  [  .   │   .   │   .   │       │  .  ]
                #  vₙ  └  [ 2.2  │  7.3  │  5.5  │ · · · │ 1.4 ]
                #              │
                #              │  (dists < protein_vdw_radius).any(dim=1)
                #              │  collapse columns: is ANY atom closer than 1.5 Å?
                #              ▼
                #
                #  protein_nearby  ── shape: (gpu_chunk_size,)  Bool
                #
                #  v₀ │ v₁ │ v₂ │ v₃ │ · · ·
                #  ─────────────────────────
                #   T  │  T  │  T  │  F  │ · · ·    T = protein overlaps voxel (blocked)
                #                                    F = voxel is free this frame
                #              │
                #              │  occupancy_gpu[ idx[protein_nearby] ] += 1
                #              │  only blocked voxels get their counter incremented
                #              ▼
                #
                #  occupancy_gpu  ── shape: (n_voxels,)  flat accumulator  [GPU]
                #
                #  ┌────┬────┬────┬────┬────┬────┐
                #  │ .. │ 3  │ 5  │ 0  │ 2  │ .. │   after N frames:
                #  └────┴────┴────┴────┴────┴────┘   value = number of frames
                #              ▲                      protein occupied this voxel
                #    idx[protein_nearby]
                #    maps blocked voxels back to their
                #    position in the full flat grid
                for i in range(0, len(shell_coords_gpu), gpu_chunk):
                    chunk = shell_coords_gpu[i:i+gpu_chunk]
                    idx   = shell_indices_gpu[i:i+gpu_chunk]
                    
                    # torch.cdist computes pairwise distances on GPU
                    # Computes all pairwise Euclidean distances between chunk voxels and protein atoms on GPU.
                    # Result shape: (chunk_size, n_protein_atoms).
                    # Each row [i] contains distances from voxel i to every protein atom.
                    dists = torch.cdist(chunk, frame_prot)

                    # Check if any protein atom within exclusion radius
                    near  = (dists < protein_vdw_radius).any(dim=1)

                    # Increment occupancy for occupied voxels
                    occupancy_gpu[idx[near]] += 1
            protein_occupancy = occupancy_gpu.cpu().numpy().reshape(nbins)
            del shell_coords_gpu, shell_indices_gpu, occupancy_gpu, frame_prot
            torch.cuda.empty_cache()
        else:
            chunk    = 5_000
            occ_flat = protein_occupancy.ravel()
            for fi in tqdm(range(trajectory.n_frames)):
                fp = protein_pos[fi]
                for i in range(0, len(shell_coords), chunk):
                    ch   = shell_coords[i:i+chunk]
                    ci   = shell_indices[i:i+chunk]
                    d    = np.linalg.norm(
                        ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                    near = (d < protein_vdw_radius).any(axis=1)
                    occ_flat[ci[near]] += 1

        # Convert counts to fractions (0 to 1)
        # fraction = count / n_frames
        # 0.0 = always free, 1.0 = always occupied
        protein_occupancy /= trajectory.n_frames

        # ── Step 5: Classify space ────────────────────────────────────────────
        print("Step 3/3: classifying space …")

        #  protein_occupancy ── fraction of frames each voxel was BLOCKED (0→1)
        #
        #  free_fraction = 1.0 - protein_occupancy
        #
        #  0.0 ◄────────────────────────────────────────────────────► 1.0
        #  always                                                   always
        #  blocked                                                   free
        #
        #  ├──────────────────┼──────────────────────┼──────────────────┤
        #  │    occupied      │      contested        │   growth space   │
        #  │   (< 0.3 free)   │   (0.3 – 0.7 free)   │   (≥ 0.7 free)   │
        #  ├──────────────────┼──────────────────────┼──────────────────┤
        #        0.0        0.3    free_fraction    0.7               1.0
        #                                         min_free_fraction
        #
        #  Each mask = in_shell  AND  threshold condition on free_fraction
        #
        #                     in_shell
        #                         │
        #          ┌──────────────┼──────────────┐
        #          ▼              ▼              ▼
        #   free < 0.3     0.3 ≤ free < 0.7   free ≥ 0.7
        #          │              │              │
        #          ▼              ▼              ▼
        #   occupied_space  contested_space  growth_space
        #      _mask             _mask           _mask
        #
        #  CROSS-SECTION VIEW (2D slice, same frame as shell diagram):
        #
        #   ┌─────────────────────────────────────┐
        #   │  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │  · outside shell (all masks False)
        #   │  ·  ·  ·  ░  ░  ▒  ▒  ░  ·  ·  ·  │
        #   │  ·  ·  ░  ░  ▒  ▒  ▒  ▒  ░  ·  ·  │
        #   │  ·  ░  ░  ▒  ▓  ▓  ▓  ▒  ░  ░  ·  │
        #   │  ·  ░  ▒  ▓  ╔══════╗  ▓  ▒  ░  ·  │  ░ growth_space     (free ≥ 0.7)
        #   │  ·  ░  ▒  ▓  ║ligand║  ▓  ▒  ░  ·  │  ▒ contested       (0.3–0.7)
        #   │  ·  ░  ▒  ▓  ╚══════╝  ▓  ▒  ░  ·  │  ▓ occupied        (free < 0.3)
        #   │  ·  ░  ░  ▒  ▓  ▓  ▓  ▒  ░  ░  ·  │
        #   │  ·  ·  ░  ░  ▒  ▒  ▒  ▒  ░  ·  ·  │
        #   │  ·  ·  ·  ░  ░  ▒  ▒  ░  ·  ·  ·  │
        #   │  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  │
        #   └─────────────────────────────────────┘
        #
        
        # Calculate fraction of time each voxel is FREE (not occupied)
        # Shape: (nbins[0], nbins[1], nbins[2]) of values 0-1
        free_fraction = 1.0 - protein_occupancy


        # Growth space = in shell AND protein-free most of the time
        # Typically use 70% threshold: "reliably free" for growth
        # Shape: (nbins[0], nbins[1], nbins[2]) of Boolean
        growth_space_mask    = in_shell & (free_fraction >= min_free_fraction)
        # growth_space_mask = in_shell & (free_fraction >= min_free_fraction) & (free_fraction != 1.0)

        # Also identify "contested space" - sometimes free, sometimes occupied
        # Useful for more aggressive modifications or flexible residues
        # Typically 30-70% free: moderate confidence
        # contested_space_mask = in_shell & (free_fraction >= 0.3) & (free_fraction < min_free_fraction)
        contested_space_mask = in_shell & (free_fraction <  min_free_fraction)
        
        # Protein-occupied space in shell
        # Usually blocked: <30% free
        occupied_space_mask  = in_shell & (protein_occupancy > 0.7)
        full_shell_mask      = in_shell

        vv = dx ** 3
        print(f"  Growth space    : {growth_space_mask.sum():,} voxels  "
              f"({growth_space_mask.sum() * vv:.1f} Å³)")
        print(f"  Contested space : {contested_space_mask.sum():,} voxels  "
              f"({contested_space_mask.sum() * vv:.1f} Å³)")
        print(f"  Occupied >70%   : {occupied_space_mask.sum():,} voxels  [diagnostic]")
        print(f"  Full shell      : {full_shell_mask.sum():,} voxels")

        result = {
            'growth_space_mask':    growth_space_mask,
            'contested_space_mask': contested_space_mask,
            'occupied_space_mask':  occupied_space_mask,
            'full_shell_mask':      full_shell_mask,
            'free_fraction':        free_fraction,
            'protein_occupancy':    protein_occupancy,
            'min_dist_to_ligand':   min_dist_to_ligand,
            'grid_info': {
                'xmin':                xmin,
                'xmax':                xmax,
                'nbins':               nbins,
                'dx':                  dx,
                'shell_inner_radius':  shell_inner_radius,
                'shell_outer_radius':  shell_outer_radius,
            },
            'volumes': {
                'growth_space': growth_space_mask.sum()    * vv,
                'contested':    contested_space_mask.sum() * vv,
                'occupied':     occupied_space_mask.sum()  * vv,
                'full_shell':   full_shell_mask.sum()      * vv,
                'voxel_size':   vv,
            },
        }
        self.negative_space_data = result
        return result

    def compute_negative_space_threaded(self,
                               shell_inner_radius: float = 1.5,
                               shell_outer_radius: float = 5.0,
                               dx: float = 0.5,
                               protein_vdw_radius: float = 1.5,
                               min_free_fraction: float = 0.7,
                               n_workers: int = None):
        """Multithreaded-CPU sibling of compute_negative_space() — same algorithm,
        same result, same GPU path when PyTorch+CUDA is available. Only the CPU
        fallback differs: instead of one serial "for fi in range(n_frames)" loop,
        the frame range is split into n_workers contiguous chunks, each processed
        by its own thread with a private accumulator; results are combined at the
        end (elementwise min for the distance map, elementwise sum for the
        occupancy counts). See compute_negative_space() for the full parameter
        docs and return value description — identical here.

        Parameters
        ----------
        n_workers : int, optional
            Number of CPU threads for the fallback path. Defaults to
            DEFAULT_THREAD_WORKERS. Ignored when GPU acceleration is active.
        """
        if not MDTRAJ_AVAILABLE:
            raise ImportError("MDTraj required — install via conda-forge: mdtraj")
        if self.prot_lig_traj is None:
            raise RuntimeError("Call load() before compute_negative_space_threaded()")

        trajectory    = self.prot_lig_traj
        ligand_atoms  = self._ligand_sel_idx
        protein_atoms = self._protein_sel_idx

        print(f"Shell: {shell_inner_radius}–{shell_outer_radius} Å | "
              f"voxel {dx} Å | excl. radius {protein_vdw_radius} Å | "
              f"free threshold {min_free_fraction*100:.0f}%"
              + ("" if USE_GPU else f" | CPU threads: "
                 f"{max(1, min(n_workers or DEFAULT_THREAD_WORKERS, trajectory.n_frames))}"))

        # ── Step 1: Grid setup (identical to compute_negative_space) ───────────
        ligand_pos  = trajectory.xyz[:, ligand_atoms,  :] * 10.0
        protein_pos = trajectory.xyz[:, protein_atoms, :] * 10.0

        all_ligand_pos = ligand_pos.reshape(-1, 3)
        xmin = all_ligand_pos.min(axis=0) - shell_outer_radius - 2.0
        xmax = all_ligand_pos.max(axis=0) + shell_outer_radius + 2.0
        nbins = np.ceil((xmax - xmin) / dx).astype(int)
        print(f"Grid: {nbins}  ({np.prod(nbins):,} voxels)")

        x, y, z      = (np.arange(nbins[i]) * dx + xmin[i] for i in range(3))
        xx, yy, zz   = np.meshgrid(x, y, z, indexing='ij')
        voxel_coords = np.stack([xx, yy, zz], axis=-1)
        voxel_flat   = voxel_coords.reshape(-1, 3)

        # ── Step 2: Minimum ligand distance per voxel ───────────────────────────
        print("Step 1/3: ligand distance map …")
        min_dist_to_ligand = np.full(nbins, np.inf, dtype=np.float32)

        if USE_GPU:
            # ---- GPU-ACCELERATED PATH (identical to compute_negative_space) ----
            voxel_flat_gpu = torch.tensor(voxel_flat, dtype=torch.float32, device=GPU_DEVICE)
            min_dist_gpu   = torch.full((len(voxel_flat),), float('inf'),
                                        dtype=torch.float32, device=GPU_DEVICE)
            gpu_chunk = 500_000
            for fi in tqdm(range(trajectory.n_frames)):
                frame_lig = torch.tensor(ligand_pos[fi], dtype=torch.float32, device=GPU_DEVICE)
                for i in range(0, len(voxel_flat_gpu), gpu_chunk):
                    chunk = voxel_flat_gpu[i:i+gpu_chunk]
                    dists = torch.cdist(chunk, frame_lig)
                    min_dist_gpu[i:i+gpu_chunk] = torch.minimum(
                        min_dist_gpu[i:i+gpu_chunk], dists.min(dim=1).values)
            min_dist_to_ligand = min_dist_gpu.cpu().numpy().reshape(nbins)
            del voxel_flat_gpu, min_dist_gpu, frame_lig
            torch.cuda.empty_cache()
        else:
            # ---- MULTITHREADED CPU PATH ----
            # Same per-frame math as compute_negative_space()'s CPU loop; frames
            # are split across threads, each with its own private running-min
            # buffer, combined with an elementwise min at the end.
            chunk = 10_000

            def _partial_min_dist(start, stop):
                local = np.full(nbins, np.inf, dtype=np.float32).ravel()
                for fi in range(start, stop):
                    fp = ligand_pos[fi]
                    for i in range(0, len(voxel_flat), chunk):
                        ch = voxel_flat[i:i+chunk]
                        d  = np.linalg.norm(
                            ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2
                        ).min(axis=1)
                        local[i:i+chunk] = np.minimum(local[i:i+chunk], d)
                return local.reshape(nbins)

            min_dist_to_ligand = _threaded_frame_reduce(
                trajectory.n_frames, _partial_min_dist,
                lambda parts: np.minimum.reduce(parts), n_workers=n_workers)

        # ── Step 3: Shell mask (identical to compute_negative_space) ───────────
        in_shell = ((min_dist_to_ligand >= shell_inner_radius) &
                    (min_dist_to_ligand <= shell_outer_radius))
        print(f"Shell voxels: {in_shell.sum():,}")

        # ── Step 4: Protein occupancy within shell ──────────────────────────────
        print("Step 2/3: protein occupancy …")
        protein_occupancy = np.zeros(nbins, dtype=np.float32)

        shell_coords  = voxel_coords[in_shell]
        shell_indices = np.where(in_shell.ravel())[0]

        if USE_GPU:
            shell_coords_gpu  = torch.tensor(shell_coords,  dtype=torch.float32, device=GPU_DEVICE)
            shell_indices_gpu = torch.tensor(shell_indices, dtype=torch.long,    device=GPU_DEVICE)
            occupancy_gpu     = torch.zeros(int(np.prod(nbins)), dtype=torch.float32, device=GPU_DEVICE)
            n_prot    = len(protein_atoms)
            gpu_chunk = max(1_000, min(200_000, int(4e9 / (n_prot * 4))))
            for fi in tqdm(range(trajectory.n_frames)):
                frame_prot = torch.tensor(protein_pos[fi], dtype=torch.float32, device=GPU_DEVICE)
                for i in range(0, len(shell_coords_gpu), gpu_chunk):
                    chunk = shell_coords_gpu[i:i+gpu_chunk]
                    idx   = shell_indices_gpu[i:i+gpu_chunk]
                    dists = torch.cdist(chunk, frame_prot)
                    near  = (dists < protein_vdw_radius).any(dim=1)
                    occupancy_gpu[idx[near]] += 1
            protein_occupancy = occupancy_gpu.cpu().numpy().reshape(nbins)
            del shell_coords_gpu, shell_indices_gpu, occupancy_gpu, frame_prot
            torch.cuda.empty_cache()
        else:
            # ---- MULTITHREADED CPU PATH ----
            # Same per-frame math as compute_negative_space()'s CPU loop; each
            # thread accumulates occupancy counts for its own frame range into a
            # private array, summed together at the end (counts are additive).
            chunk = 5_000

            def _partial_occupancy(start, stop):
                local = np.zeros(nbins, dtype=np.float32)
                local_flat = local.ravel()
                for fi in range(start, stop):
                    fp = protein_pos[fi]
                    for i in range(0, len(shell_coords), chunk):
                        ch   = shell_coords[i:i+chunk]
                        ci   = shell_indices[i:i+chunk]
                        d    = np.linalg.norm(
                            ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                        near = (d < protein_vdw_radius).any(axis=1)
                        local_flat[ci[near]] += 1
                return local

            protein_occupancy = _threaded_frame_reduce(
                trajectory.n_frames, _partial_occupancy, _sum_reduce, n_workers=n_workers)

        protein_occupancy /= trajectory.n_frames

        # ── Step 5: Classify space (identical to compute_negative_space) ───────
        print("Step 3/3: classifying space …")
        free_fraction = 1.0 - protein_occupancy
        growth_space_mask    = in_shell & (free_fraction >= min_free_fraction)
        contested_space_mask = in_shell & (free_fraction <  min_free_fraction)
        occupied_space_mask  = in_shell & (protein_occupancy > 0.7)
        full_shell_mask      = in_shell

        vv = dx ** 3
        print(f"  Growth space    : {growth_space_mask.sum():,} voxels  "
              f"({growth_space_mask.sum() * vv:.1f} Å³)")
        print(f"  Contested space : {contested_space_mask.sum():,} voxels  "
              f"({contested_space_mask.sum() * vv:.1f} Å³)")
        print(f"  Occupied >70%   : {occupied_space_mask.sum():,} voxels  [diagnostic]")
        print(f"  Full shell      : {full_shell_mask.sum():,} voxels")

        result = {
            'growth_space_mask':    growth_space_mask,
            'contested_space_mask': contested_space_mask,
            'occupied_space_mask':  occupied_space_mask,
            'full_shell_mask':      full_shell_mask,
            'free_fraction':        free_fraction,
            'protein_occupancy':    protein_occupancy,
            'min_dist_to_ligand':   min_dist_to_ligand,
            'grid_info': {
                'xmin':                xmin,
                'xmax':                xmax,
                'nbins':               nbins,
                'dx':                  dx,
                'shell_inner_radius':  shell_inner_radius,
                'shell_outer_radius':  shell_outer_radius,
            },
            'volumes': {
                'growth_space': growth_space_mask.sum()    * vv,
                'contested':    contested_space_mask.sum() * vv,
                'occupied':     occupied_space_mask.sum()  * vv,
                'full_shell':   full_shell_mask.sum()      * vv,
                'voxel_size':   vv,
            },
        }
        self.negative_space_data = result
        return result

    def compute_growth_space_features(self,
                                      aromatic_cutoff: float = 4.5,
                                      hbond_cutoff: float = 3.5,
                                      hydrophobic_cutoff: float = 5.0):
        """Score each growth-space voxel by proximity to protein chemical feature types.

        For each voxel in the growth space (protein occupancy < 30%), computes a
        linear proximity score to six protein atom categories per trajectory frame,
        then averages over all frames:
            score = 1 − (min_distance / cutoff)   if min_distance < cutoff
            score = 0                               otherwise

        Protein atom categories:
          - aromatic        : ring atoms of TYR, PHE, TRP, HIS
          - hydrophobic     : side-chain C of ALA, VAL, LEU, ILE, MET, PHE, TRP, PRO
          - hbond_donor     : near protein H-bond acceptors (suggest donor in ligand)
          - hbond_acceptor  : near protein H-bond donors   (suggest acceptor in ligand)
          - positive_charge : near protein negative charges (ASP/GLU carboxylates)
          - negative_charge : near protein positive charges (LYS/ARG/HIS)

        Charge score arrays remain zero if the protein has no charged residues
        of that sign (common in IDPs — no failure).

        GPU-accelerated when PyTorch + CUDA are available; falls back to NumPy/CPU.

        Requires compute_negative_space() to have been called first.

        Parameters
        ----------
        aromatic_cutoff : float
            Distance cutoff for aromatic interactions (Å). Default 4.5 Å.
        hbond_cutoff : float
            Distance cutoff for H-bond and charge interactions (Å). Default 3.5 Å.
        hydrophobic_cutoff : float
            Distance cutoff for hydrophobic contacts (Å). Default 5.0 Å.

        Returns
        -------
        dict with keys:
          aromatic_score        : (nx,ny,nz) float
          hydrophobic_score     : (nx,ny,nz) float
          hbond_donor_score     : (nx,ny,nz) float  (high near protein H-bond acceptors)
          hbond_acceptor_score  : (nx,ny,nz) float  (high near protein H-bond donors)
          positive_charge_score : (nx,ny,nz) float  (high near protein negative charges)
          negative_charge_score : (nx,ny,nz) float  (high near protein positive charges)
          grid_info             : dict
          feature_names         : list[str]
        Stored on self.growth_space_features.
        """
        if self.negative_space_data is None:
            raise RuntimeError(
                "Call compute_negative_space() before compute_growth_space_features()"
            )

        trajectory  = self.prot_lig_traj
        grid_info   = self.negative_space_data['grid_info']
        xmin        = grid_info['xmin']
        dx          = grid_info['dx']
        nbins       = grid_info['nbins']
        growth_mask = self.negative_space_data['growth_space_mask']

        # ========================================================================
        # STEP 1: IDENTIFY AND CATEGORIZE PROTEIN ATOMS BY CHEMICAL TYPE
        # ========================================================================
        # Six categories, mirroring the six score arrays returned below:
        #   aromatic          — ring atoms of TRP/TYR/PHE/HIS (π-π / CH-π contacts)
        #   hydrophobic       — nonpolar side-chain carbons (van der Waals contacts)
        #   hbond_donor_atoms / hbond_acceptor_atoms — protein groups that can
        #       give/receive an H-bond; scored in reverse (see Step 4)
        #   positive_atoms / negative_atoms — salt-bridge partners; scored in
        #       reverse (see Step 5)
        # ── Atom selections ( all indices in prot_lig context) ─────────────────
        
        # Aromatic ring atoms via get_protein_rings() rather than a hardcoded
        # atom-name list: it selects the correct ring atoms per residue type
        # (including ring nitrogens — HIS ND1/NE2, TRP NE1), so no ring atom
        # is missed the way a single shared name list would.
        protein_rings, _, _ = self.get_protein_rings()
        aromatic_atoms = (np.concatenate(protein_rings).astype(int)
                          if protein_rings else np.array([], dtype=int))

        # Hydrophobic side-chain carbons (selective: named carbons in nonpolar residues)
        hydrophobic_atoms = self.prot_lig_top.select(
            'protein and (resname ALA VAL LEU ILE MET PHE TRP PRO) '
            'and (name CB CG CG1 CG2 CD CD1 CD2 CE CE1 CE2 CZ)'
        )

        # H-bond donors (protein side): backbone/sidechain N (has an N-H to give)
        # plus hydroxyl O in SER/THR/TYR/ASN/GLN. These atoms are scored against
        # hbond_acceptor_score, NOT hbond_donor_score — see Step 4.
        hbond_donor_atoms = np.union1d(
            self.prot_lig_top.select('protein and element N'),
            self.prot_lig_top.select(
                'protein and element O and resname SER THR TYR ASN GLN'
            ),
        )

        # H-bond acceptors (protein side): every O/N has a lone pair, so no
        # residue filter is needed here (unlike donors, which need the -OH/-NH
        # check). Scored against hbond_donor_score — see Step 4.
        hbond_acceptor_atoms = self.prot_lig_top.select(
            'protein and (element O or element N)'
        )

        # Charged atoms: only the specific side-chain atom that carries the
        # charge (LYS NZ, ARG NH1/NH2, HIS NE2 / ASP OD1/OD2, GLU OE1/OE2).
        # Either array can be empty for an IDP lacking that charge type —
        # the corresponding score array is simply left at all-zero (Step 6).
        positive_atoms = self.prot_lig_top.select(
            'protein and resname LYS ARG HIS and name NZ NH1 NH2 NE2'
        )
        negative_atoms = self.prot_lig_top.select(
            'protein and resname ASP GLU and name OD1 OD2 OE1 OE2'
        )

        print(f"  Aromatic ring atoms   : {len(aromatic_atoms)}")
        print(f"  Hydrophobic C atoms   : {len(hydrophobic_atoms)}")
        print(f"  H-bond donor atoms    : {len(hbond_donor_atoms)}")
        print(f"  H-bond acceptor atoms : {len(hbond_acceptor_atoms)}")
        print(f"  Positive charge atoms : {len(positive_atoms)}"
              + (" [none — score stays zero]" if len(positive_atoms) == 0 else ""))
        print(f"  Negative charge atoms : {len(negative_atoms)}"
              + (" [none — score stays zero]" if len(negative_atoms) == 0 else ""))

        # ========================================================================
        # STEP 2: RECREATE THE 3D VOXEL GRID AND EXTRACT GROWTH-SPACE VOXELS
        # ========================================================================
        # Same grid definition used in compute_negative_space(): voxel centers
        # spaced dx apart starting at xmin. growth_voxel_coords holds only the
        # coordinates of voxels flagged True in growth_mask; growth_indices are
        # their flat (ravelled) positions, used to scatter scores back into the
        # full (nx,ny,nz) array without ever materializing a full dense grid.
        # ── Voxel grid for growth space ────────────────────────────────────────
        x, y, z      = (np.arange(nbins[i]) * dx + xmin[i] for i in range(3))
        xx, yy, zz   = np.meshgrid(x, y, z, indexing='ij')
        voxel_coords = np.stack([xx, yy, zz], axis=-1)

        growth_voxel_coords = voxel_coords[growth_mask]          # (n_growth, 3)
        growth_indices      = np.where(growth_mask.ravel())[0]   # flat indices into ravelled grid

        print(f"\nScoring {len(growth_voxel_coords):,} growth-space voxels …")

        # ========================================================================
        # STEP 3: INITIALIZE FEATURE SCORE FIELDS
        # ========================================================================
        # Full-grid arrays, zero everywhere except growth-space voxels that end
        # up within cutoff of a protein atom of that type. Convention:
        #   0.0 = no atom of this type within cutoff, or voxel not in growth space
        #   1.0 = an atom of this type sits right at the voxel center
        #   otherwise = 1.0 − (min_distance_to_nearest_atom / cutoff)
        # ── Initialize score arrays (zeros = no feature / out of cutoff) ──────
        aromatic_score        = np.zeros(nbins, dtype=np.float32)
        hydrophobic_score     = np.zeros(nbins, dtype=np.float32)
        hbond_donor_score     = np.zeros(nbins, dtype=np.float32)
        hbond_acceptor_score  = np.zeros(nbins, dtype=np.float32)
        positive_charge_score = np.zeros(nbins, dtype=np.float32)
        negative_charge_score = np.zeros(nbins, dtype=np.float32)

        n_frames = trajectory.n_frames
        n_voxels = int(np.prod(nbins))

        # ── GPU tensors shared across all feature loops ────────────────────────
        if USE_GPU:
            voxels_gpu     = torch.tensor(
                growth_voxel_coords, dtype=torch.float32, device=GPU_DEVICE)
            growth_idx_gpu = torch.tensor(
                growth_indices, dtype=torch.long, device=GPU_DEVICE)

        # ========================================================================
        # STEP 4: SHARED SCORING KERNEL (GPU / CPU)
        # ========================================================================
        # Same linear-decay scoring used for all six feature types — factored
        # into one helper instead of repeating it per category (as the legacy
        # version did) since the only thing that changes per call is which atom
        # indices and cutoff are passed in.
        #
        #  score = 1.0 − (min_distance / cutoff)      if min_distance < cutoff
        #  score = 0.0                                  if min_distance ≥ cutoff
        #
        #  score
        #   1.0 ┤╲
        #       │  ╲
        #   0.5 ┤    ╲
        #       │      ╲
        #   0.0 ┤────────╲──────────────────────────── distance (Å)
        #       0        cutoff
        #
        # For every negative/growth-space voxel and every frame: find the
        # nearest atom of the given category, score it, and accumulate; the
        # final /n_frames average gives the trajectory-averaged suggestion
        # strength at that voxel (Step 3 in the legacy version's per-feature
        # comments — here it happens once, generically, per call).
        # ── Scoring helpers (close over shared state) ─────────────────────────
        def _score_gpu(atom_indices, cutoff):
            accum      = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            n_atoms    = len(atom_indices)
            chunk_size = max(1_000, min(200_000, int(4e9 / (n_atoms * 4))))
            for fi in tqdm(range(n_frames)):
                frame_atoms = torch.tensor(
                    trajectory.xyz[fi, atom_indices, :] * 10.0,
                    dtype=torch.float32, device=GPU_DEVICE,
                )
                for i in range(0, len(voxels_gpu), chunk_size):
                    chunk    = voxels_gpu[i:i+chunk_size]
                    idx      = growth_idx_gpu[i:i+chunk_size]
                    dists    = torch.cdist(chunk, frame_atoms)
                    min_d, _ = torch.min(dists, dim=1)
                    within   = min_d <= cutoff
                    scores   = torch.zeros(len(chunk), dtype=torch.float32, device=GPU_DEVICE)
                    scores[within] = 1.0 - min_d[within] / cutoff
                    accum[idx] += scores
                del frame_atoms
            out = accum.cpu().numpy().reshape(nbins) / n_frames
            del accum
            torch.cuda.empty_cache()
            return out

        def _score_cpu(atom_indices, cutoff):
            pos_all    = trajectory.xyz[:, atom_indices, :] * 10.0  # (n_frames, n_atoms, 3) Å
            accum      = np.zeros(nbins, dtype=np.float32)
            chunk_size = 5_000
            for fi in tqdm(range(n_frames)):
                fp = pos_all[fi]
                for i in range(0, len(growth_voxel_coords), chunk_size):
                    ch   = growth_voxel_coords[i:i+chunk_size]
                    ci   = growth_indices[i:i+chunk_size]
                    d    = np.linalg.norm(
                        ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                    md   = d.min(axis=1)
                    w    = md <= cutoff
                    s    = np.zeros(len(ch), dtype=np.float32)
                    s[w] = 1.0 - md[w] / cutoff
                    accum.ravel()[ci] += s
            return accum / n_frames

        _score = _score_gpu if USE_GPU else _score_cpu

        # ========================================================================
        # STEP 5: SCORE EACH FEATURE TYPE
        # ========================================================================
        # aromatic / hydrophobic are scored directly (protein feature → same
        # ligand feature suggested at that voxel).
        #
        # hbond and charge scoring is COMPLEMENTARY — the protein group present
        # determines the OPPOSITE ligand group to suggest:
        #   protein H-bond donor    → suggest ligand H-bond ACCEPTOR here
        #   protein H-bond acceptor → suggest ligand H-bond DONOR here
        #   protein positive charge → suggest ligand NEGATIVE group here (salt bridge)
        #   protein negative charge → suggest ligand POSITIVE group here (salt bridge)
        # ── Per-feature scoring ────────────────────────────────────────────────
        if len(aromatic_atoms) > 0:
            print("Aromatic …")
            aromatic_score = _score(aromatic_atoms, aromatic_cutoff)

        if len(hydrophobic_atoms) > 0:
            print("Hydrophobic …")
            hydrophobic_score = _score(hydrophobic_atoms, hydrophobic_cutoff)

        if len(hbond_donor_atoms) > 0:
            # Protein donors → suggest ligand acceptor at this location
            print("H-bond donors (→ acceptor score) …")
            hbond_acceptor_score = _score(hbond_donor_atoms, hbond_cutoff)

        if len(hbond_acceptor_atoms) > 0:
            # Protein acceptors → suggest ligand donor at this location
            print("H-bond acceptors (→ donor score) …")
            hbond_donor_score = _score(hbond_acceptor_atoms, hbond_cutoff)

        if len(positive_atoms) > 0:
            # Protein positive charges → suggest negative ligand group
            print("Positive charges (→ negative charge score) …")
            negative_charge_score = _score(positive_atoms, hbond_cutoff)

        if len(negative_atoms) > 0:
            # Protein negative charges → suggest positive ligand group
            print("Negative charges (→ positive charge score) …")
            positive_charge_score = _score(negative_atoms, hbond_cutoff)

        # GPU cleanup — unconditional to avoid tensor leaks if charge atoms are absent
        if USE_GPU:
            del voxels_gpu, growth_idx_gpu
            torch.cuda.empty_cache()

        # ========================================================================
        # STEP 6: PACKAGE RESULTS
        # ========================================================================
        # Charge/hbond score arrays for a missing category stay all-zero
        # (initialized in Step 3, never written to) rather than raising —
        # expected for IDPs that may lack, e.g., any negatively charged residue.
        result = {
            'aromatic_score':        aromatic_score,
            'hydrophobic_score':     hydrophobic_score,
            'hbond_donor_score':     hbond_donor_score,
            'hbond_acceptor_score':  hbond_acceptor_score,
            'positive_charge_score': positive_charge_score,
            'negative_charge_score': negative_charge_score,
            'grid_info':             grid_info,
            'feature_names': [
                'Aromatic', 'Hydrophobic',
                'H-bond Donor', 'H-bond Acceptor',
                'Positive Charge', 'Negative Charge',
            ],
        }
        self.growth_space_features = result
        return result

    def compute_growth_space_features_threaded(self,
                                      aromatic_cutoff: float = 4.5,
                                      hbond_cutoff: float = 3.5,
                                      hydrophobic_cutoff: float = 5.0,
                                      n_workers: int = None):
        """Multithreaded-CPU sibling of compute_growth_space_features() — same
        algorithm, same result, same GPU path when PyTorch+CUDA is available.
        Only the CPU scoring kernel (_score_cpu) differs: frames are split into
        n_workers contiguous ranges, each scored by its own thread into a
        private accumulator, then summed together (scores are additive before
        the final /n_frames average). See compute_growth_space_features() for
        the full parameter docs and return value description — identical here.

        Parameters
        ----------
        n_workers : int, optional
            Number of CPU threads for the fallback path. Defaults to
            DEFAULT_THREAD_WORKERS. Ignored when GPU acceleration is active.
        """
        if self.negative_space_data is None:
            raise RuntimeError(
                "Call compute_negative_space() before compute_growth_space_features_threaded()"
            )

        trajectory  = self.prot_lig_traj
        grid_info   = self.negative_space_data['grid_info']
        xmin        = grid_info['xmin']
        dx          = grid_info['dx']
        nbins       = grid_info['nbins']
        growth_mask = self.negative_space_data['growth_space_mask']

        # ── Atom selections (identical to compute_growth_space_features) ───────
        protein_rings, _, _ = self.get_protein_rings()
        aromatic_atoms = (np.concatenate(protein_rings).astype(int)
                          if protein_rings else np.array([], dtype=int))

        hydrophobic_atoms = self.prot_lig_top.select(
            'protein and (resname ALA VAL LEU ILE MET PHE TRP PRO) '
            'and (name CB CG CG1 CG2 CD CD1 CD2 CE CE1 CE2 CZ)'
        )
        hbond_donor_atoms = np.union1d(
            self.prot_lig_top.select('protein and element N'),
            self.prot_lig_top.select(
                'protein and element O and resname SER THR TYR ASN GLN'
            ),
        )
        hbond_acceptor_atoms = self.prot_lig_top.select(
            'protein and (element O or element N)'
        )
        positive_atoms = self.prot_lig_top.select(
            'protein and resname LYS ARG HIS and name NZ NH1 NH2 NE2'
        )
        negative_atoms = self.prot_lig_top.select(
            'protein and resname ASP GLU and name OD1 OD2 OE1 OE2'
        )

        print(f"  Aromatic ring atoms   : {len(aromatic_atoms)}")
        print(f"  Hydrophobic C atoms   : {len(hydrophobic_atoms)}")
        print(f"  H-bond donor atoms    : {len(hbond_donor_atoms)}")
        print(f"  H-bond acceptor atoms : {len(hbond_acceptor_atoms)}")
        print(f"  Positive charge atoms : {len(positive_atoms)}"
              + (" [none — score stays zero]" if len(positive_atoms) == 0 else ""))
        print(f"  Negative charge atoms : {len(negative_atoms)}"
              + (" [none — score stays zero]" if len(negative_atoms) == 0 else ""))
        if not USE_GPU:
            print(f"  CPU threads           : "
                  f"{max(1, min(n_workers or DEFAULT_THREAD_WORKERS, trajectory.n_frames))}")

        # ── Voxel grid for growth space (identical) ─────────────────────────────
        x, y, z      = (np.arange(nbins[i]) * dx + xmin[i] for i in range(3))
        xx, yy, zz   = np.meshgrid(x, y, z, indexing='ij')
        voxel_coords = np.stack([xx, yy, zz], axis=-1)

        growth_voxel_coords = voxel_coords[growth_mask]
        growth_indices      = np.where(growth_mask.ravel())[0]

        print(f"\nScoring {len(growth_voxel_coords):,} growth-space voxels …")

        aromatic_score        = np.zeros(nbins, dtype=np.float32)
        hydrophobic_score     = np.zeros(nbins, dtype=np.float32)
        hbond_donor_score     = np.zeros(nbins, dtype=np.float32)
        hbond_acceptor_score  = np.zeros(nbins, dtype=np.float32)
        positive_charge_score = np.zeros(nbins, dtype=np.float32)
        negative_charge_score = np.zeros(nbins, dtype=np.float32)

        n_frames = trajectory.n_frames
        n_voxels = int(np.prod(nbins))

        if USE_GPU:
            voxels_gpu     = torch.tensor(
                growth_voxel_coords, dtype=torch.float32, device=GPU_DEVICE)
            growth_idx_gpu = torch.tensor(
                growth_indices, dtype=torch.long, device=GPU_DEVICE)

        def _score_gpu(atom_indices, cutoff):
            accum      = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            n_atoms    = len(atom_indices)
            chunk_size = max(1_000, min(200_000, int(4e9 / (n_atoms * 4))))
            for fi in tqdm(range(n_frames)):
                frame_atoms = torch.tensor(
                    trajectory.xyz[fi, atom_indices, :] * 10.0,
                    dtype=torch.float32, device=GPU_DEVICE,
                )
                for i in range(0, len(voxels_gpu), chunk_size):
                    chunk    = voxels_gpu[i:i+chunk_size]
                    idx      = growth_idx_gpu[i:i+chunk_size]
                    dists    = torch.cdist(chunk, frame_atoms)
                    min_d, _ = torch.min(dists, dim=1)
                    within   = min_d <= cutoff
                    scores   = torch.zeros(len(chunk), dtype=torch.float32, device=GPU_DEVICE)
                    scores[within] = 1.0 - min_d[within] / cutoff
                    accum[idx] += scores
                del frame_atoms
            out = accum.cpu().numpy().reshape(nbins) / n_frames
            del accum
            torch.cuda.empty_cache()
            return out

        def _score_cpu_threaded(atom_indices, cutoff):
            # Multithreaded version of compute_growth_space_features()'s
            # _score_cpu: frames split across threads, each with a private
            # accumulator (scores are additive), summed at the end.
            pos_all    = trajectory.xyz[:, atom_indices, :] * 10.0  # (n_frames, n_atoms, 3) Å
            chunk_size = 5_000

            def _partial(start, stop):
                local = np.zeros(nbins, dtype=np.float32)
                local_flat = local.ravel()
                for fi in range(start, stop):
                    fp = pos_all[fi]
                    for i in range(0, len(growth_voxel_coords), chunk_size):
                        ch   = growth_voxel_coords[i:i+chunk_size]
                        ci   = growth_indices[i:i+chunk_size]
                        d    = np.linalg.norm(
                            ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                        md   = d.min(axis=1)
                        w    = md <= cutoff
                        s    = np.zeros(len(ch), dtype=np.float32)
                        s[w] = 1.0 - md[w] / cutoff
                        local_flat[ci] += s
                return local

            accum = _threaded_frame_reduce(n_frames, _partial, _sum_reduce, n_workers=n_workers)
            return accum / n_frames

        _score = _score_gpu if USE_GPU else _score_cpu_threaded

        if len(aromatic_atoms) > 0:
            print("Aromatic …")
            aromatic_score = _score(aromatic_atoms, aromatic_cutoff)

        if len(hydrophobic_atoms) > 0:
            print("Hydrophobic …")
            hydrophobic_score = _score(hydrophobic_atoms, hydrophobic_cutoff)

        if len(hbond_donor_atoms) > 0:
            print("H-bond donors (→ acceptor score) …")
            hbond_acceptor_score = _score(hbond_donor_atoms, hbond_cutoff)

        if len(hbond_acceptor_atoms) > 0:
            print("H-bond acceptors (→ donor score) …")
            hbond_donor_score = _score(hbond_acceptor_atoms, hbond_cutoff)

        if len(positive_atoms) > 0:
            print("Positive charges (→ negative charge score) …")
            negative_charge_score = _score(positive_atoms, hbond_cutoff)

        if len(negative_atoms) > 0:
            print("Negative charges (→ positive charge score) …")
            positive_charge_score = _score(negative_atoms, hbond_cutoff)

        if USE_GPU:
            del voxels_gpu, growth_idx_gpu
            torch.cuda.empty_cache()

        result = {
            'aromatic_score':        aromatic_score,
            'hydrophobic_score':     hydrophobic_score,
            'hbond_donor_score':     hbond_donor_score,
            'hbond_acceptor_score':  hbond_acceptor_score,
            'positive_charge_score': positive_charge_score,
            'negative_charge_score': negative_charge_score,
            'grid_info':             grid_info,
            'feature_names': [
                'Aromatic', 'Hydrophobic',
                'H-bond Donor', 'H-bond Acceptor',
                'Positive Charge', 'Negative Charge',
            ],
        }
        self.growth_space_features = result
        return result

    # ── 5. Pharmacophore definition ───────────────────────────────────────────

    def compute_pharmacophore_maps(self, protein_vdw_radius: float = 1.5):
        """Compute residue-type protein occupancy maps for pharmacophore definition.

        For each voxel in the analysis shell, accumulates per-frame binary
        occupancy (is an atom of this chemical type within protein_vdw_radius?)
        and averages over the trajectory. The algorithm runs once on the full
        shell and derives the contested-space result by masking.

        Six residue-type occupancy maps are produced:
          - aromatic         : ring-center occupancy (with contact/non-contact split
                               when compute_aromatic_contacts() has been called)
          - hydrophobic      : side-chain C atoms of nonpolar residues
          - hbond_donors     : backbone/sidechain N + polar hydroxyl O
          - hbond_acceptors  : all protein O and N
          - positive_charge  : LYS/ARG/HIS charged N atoms
          - negative_charge  : ASP/GLU carboxylate O atoms

        Aromatic occupancy uses per-frame ring centroids (not individual atoms)
        and is split into:
          - aromatic_occupancy_contacts     : voxel blocked by a ring in active
                                             ligand contact that frame
          - aromatic_occupancy_non_contacts : voxel blocked by a ring NOT in
                                             active contact (transient passage)
        Requires compute_aromatic_contacts() for the contact split; gracefully
        sets both to None if that method has not been called.

        Charged-residue maps remain all-zero if no such residues exist (IDP
        fallback — no failure).

        GPU-accelerated when PyTorch + CUDA are available; falls back to
        NumPy/CPU.

        Requires compute_negative_space() to have been called first.

        Parameters
        ----------
        protein_vdw_radius : float
            Atom exclusion radius used to decide voxel occupancy (Å).
            Default 1.5 Å — matches compute_negative_space() convention.

        Returns
        -------
        (maps_contested, maps_full) — two dicts with identical keys:
          aromatic_occupancy              : (nx,ny,nz) float
          aromatic_occupancy_contacts     : (nx,ny,nz) float or None
          aromatic_occupancy_non_contacts : (nx,ny,nz) float or None
          hydrophobic_occupancy           : (nx,ny,nz) float
          hbond_donors_occupancy          : (nx,ny,nz) float
          hbond_acceptors_occupancy       : (nx,ny,nz) float
          positive_occupancy              : (nx,ny,nz) float
          negative_occupancy              : (nx,ny,nz) float
          grid_info                       : dict
        Stored on self.pharmacophore_maps_contested and self.pharmacophore_maps_full.
        """
        if self.negative_space_data is None:
            raise RuntimeError(
                "Call compute_negative_space() before compute_pharmacophore_maps()"
            )

        # ========================================================================
        # STEP 1: EXTRACT GRID INFORMATION
        # ========================================================================
        # Grid parameters and both masks come from compute_negative_space():
        # full_shell_mask defines every voxel the algorithm scores; contested_mask
        # is a subset of it. Scoring runs ONCE over the full shell — the
        # contested-space result is derived afterward by zeroing everything
        # outside contested_mask (see _mask_to_contested below), instead of
        # repeating the whole scoring pass a second time over a smaller mask.
        trajectory        = self.prot_lig_traj
        grid_info         = self.negative_space_data['grid_info']
        xmin              = grid_info['xmin']
        dx                = grid_info['dx']
        nbins             = grid_info['nbins']
        full_shell_mask   = self.negative_space_data['full_shell_mask']
        contested_mask    = self.negative_space_data['contested_space_mask']

        n_frames = trajectory.n_frames
        n_voxels = int(np.prod(nbins))

        # ========================================================================
        # STEP 2: IDENTIFY PROTEIN ATOMS BY CHEMICAL TYPE + AROMATIC RING GROUPS
        # ========================================================================
        # Same six categories as the legacy version (aromatic ring atoms,
        # hydrophobic side-chain carbons, H-bond donors/acceptors, +/- charges),
        # but built from get_protein_rings()/MDTraj select() strings instead of
        # a manual per-residue loop over topology.residues — see the earlier
        # comparison of get_protein_rings() vs. the hardcoded ring atom names.
        # ── Atom selections (prot_lig context) ────────────────────────────────
        protein_rings, protein_rings_index, _ = self.get_protein_rings()

        # Ring CENTROIDS, not individual ring atoms: for each aromatic residue,
        # average the position of its ring atoms every frame. One centroid per
        # residue per frame — this is what gets tested for proximity to a voxel,
        # exactly like the legacy ring_centers_pos array.
        # Per-frame ring centroids: (n_frames, n_aro_residues, 3) in Å
        n_aro_residues = len(protein_rings)
        ring_centers_pos = np.zeros((n_frames, n_aro_residues, 3), dtype=np.float32)
        for res_idx, atom_indices in enumerate(protein_rings):
            ring_centers_pos[:, res_idx, :] = (
                trajectory.xyz[:, atom_indices, :].mean(axis=1) * 10.0
            )

        # Per-residue, per-frame contact flags — the legacy version read these
        # from a hardcoded pickle path (config["aromatic_contacts_pickle"]);
        # here they come directly from self._stacked_full, populated by
        # compute_aromatic_contacts() earlier in the pipeline. If that method
        # was never called, the contact/non-contact split is skipped rather
        # than failing (there is no separate pickle to fall back to).
        # Aromatic contact flags: (n_frames, n_aro_residues) — from compute_aromatic_contacts()
        # Columns ordered identically to protein_rings / ring_centers_pos.
        have_aro_contacts = (self._stacked_full is not None
                             and len(protein_rings_index) > 0)
        if have_aro_contacts:
            aromatic_contacts = self._stacked_full[:, self._protein_rings_index].astype(bool)
        else:
            aromatic_contacts = None
            print("  compute_aromatic_contacts() not called — contact/non-contact "
                  "split will be skipped")

        # Same optional-split idea, but for H-bond donors/acceptors instead of
        # aromatics — the legacy version had no equivalent for H-bonds at all;
        # this is new behavior enabled by compute_hbond_contacts() upstream.
        have_hbond_contacts = (self.hbond_contact_frames_pd is not None and
                               self.hbond_contact_frames_ld is not None)
        if not have_hbond_contacts:
            print("  compute_hbond_contacts() not called — H-bond contact/non-contact "
                  "split will be skipped")

        hydrophobic_atoms = self.prot_lig_top.select(
            'protein and (resname ALA VAL LEU ILE MET PHE TRP PRO) '
            'and (name CB CG CG1 CG2 CD CD1 CD2 CE CE1 CE2 CZ)'
        )
        hbond_donor_atoms = np.union1d(
            self.prot_lig_top.select('protein and element N'),
            self.prot_lig_top.select(
                'protein and element O and resname SER THR TYR ASN GLN'
            ),
        )
        hbond_acceptor_atoms = self.prot_lig_top.select(
            'protein and (element O or element N)'
        )
        positive_atoms = self.prot_lig_top.select(
            'protein and resname LYS ARG HIS and name NZ NH1 NH2 NE2'
        )
        negative_atoms = self.prot_lig_top.select(
            'protein and resname ASP GLU and name OD1 OD2 OE1 OE2'
        )

        print(f"  Aromatic ring groups  : {n_aro_residues}")
        print(f"  Hydrophobic C atoms   : {len(hydrophobic_atoms)}")
        print(f"  H-bond donor atoms    : {len(hbond_donor_atoms)}")
        print(f"  H-bond acceptor atoms : {len(hbond_acceptor_atoms)}")
        print(f"  Positive charge atoms : {len(positive_atoms)}"
              + (" [none — score stays zero]" if len(positive_atoms) == 0 else ""))
        print(f"  Negative charge atoms : {len(negative_atoms)}"
              + (" [none — score stays zero]" if len(negative_atoms) == 0 else ""))

        # Atom-to-residue index arrays for the H-bond contact split.
        # residue.index in prot_lig_top for protein-only selections falls in
        # [0, n_protein_residues-1], matching the column space of
        # hbond_contact_frames_pd / hbond_contact_frames_ld.
        donor_atom_to_res = np.array(
            [self.prot_lig_top.atom(a).residue.index for a in hbond_donor_atoms],
            dtype=int,
        )
        acceptor_atom_to_res = np.array(
            [self.prot_lig_top.atom(a).residue.index for a in hbond_acceptor_atoms],
            dtype=int,
        )

        # ========================================================================
        # STEP 3: CREATE 3D VOXEL GRID (FULL SHELL, NOT JUST CONTESTED)
        # ========================================================================
        # Same grid reconstruction as the legacy version's Step 3, but built
        # over full_shell_mask rather than contested_mask directly — scoring
        # the whole shell once and masking down to contested afterward avoids
        # ever needing a second full pass restricted to contested voxels.
        # ── Voxel grid — full shell (contested is a subset, derived by masking) ─
        x, y, z      = (np.arange(nbins[i]) * dx + xmin[i] for i in range(3))
        xx, yy, zz   = np.meshgrid(x, y, z, indexing='ij')
        voxel_coords = np.stack([xx, yy, zz], axis=-1)

        shell_voxel_coords = voxel_coords[full_shell_mask]
        shell_indices      = np.where(full_shell_mask.ravel())[0]

        print(f"\nScoring {len(shell_voxel_coords):,} shell voxels "
              f"(full shell — contested derived by masking) …")

        # ========================================================================
        # STEP 4: INITIALIZE OCCUPANCY ACCUMULATORS
        # ========================================================================
        # Every _occ array is 0.0 by construction (full-shell shape) — a voxel
        # only becomes nonzero if a protein atom of that type sits within
        # protein_vdw_radius of it on at least one frame. The *_contacts_occ /
        # *_non_contacts_occ arrays are left as None (not zero arrays) when the
        # corresponding compute_*_contacts() step wasn't run, so downstream code
        # can distinguish "never computed" from "computed, always zero".
        # ── Initialize accumulators (full shell only) ──────────────────────────
        aromatic_occ      = np.zeros(nbins, dtype=np.float32)
        aromatic_contacts_occ     = (np.zeros(nbins, dtype=np.float32)
                                     if have_aro_contacts else None)
        aromatic_non_contacts_occ = (np.zeros(nbins, dtype=np.float32)
                                     if have_aro_contacts else None)
        hydrophobic_occ   = np.zeros(nbins, dtype=np.float32)
        hbond_donors_occ              = np.zeros(nbins, dtype=np.float32)
        hbond_donors_contacts_occ     = (np.zeros(nbins, dtype=np.float32)
                                         if have_hbond_contacts else None)
        hbond_donors_non_contacts_occ = (np.zeros(nbins, dtype=np.float32)
                                         if have_hbond_contacts else None)
        hbond_acceptors_occ              = np.zeros(nbins, dtype=np.float32)
        hbond_acceptors_contacts_occ     = (np.zeros(nbins, dtype=np.float32)
                                            if have_hbond_contacts else None)
        hbond_acceptors_non_contacts_occ = (np.zeros(nbins, dtype=np.float32)
                                            if have_hbond_contacts else None)
        positive_occ      = np.zeros(nbins, dtype=np.float32)
        negative_occ      = np.zeros(nbins, dtype=np.float32)

        # ── GPU shared tensors ─────────────────────────────────────────────────
        if USE_GPU:
            voxels_gpu     = torch.tensor(
                shell_voxel_coords, dtype=torch.float32, device=GPU_DEVICE)
            shell_idx_gpu  = torch.tensor(
                shell_indices, dtype=torch.long, device=GPU_DEVICE)

        # ========================================================================
        # STEP 5: BINARY OCCUPANCY KERNEL (no contact/non-contact split)
        # ========================================================================
        # For every shell voxel and every frame: is ANY atom of this category
        # within protein_vdw_radius? If yes, +1 to that voxel's accumulator.
        # After all frames, dividing by n_frames turns the count into an
        # occupancy FRACTION (0 → never occupied, 1 → occupied every frame) —
        # same accumulate-then-divide pattern as the legacy occupancy_gpu.
        # ── Binary occupancy helper: non-aromatic features ─────────────────────
        def _occ_gpu(atom_indices):
            accum      = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            n_atoms    = len(atom_indices)
            chunk_size = max(1_000, min(200_000, int(4e9 / (n_atoms * 4))))
            for fi in tqdm(range(n_frames)):
                frame_atoms = torch.tensor(
                    trajectory.xyz[fi, atom_indices, :] * 10.0,
                    dtype=torch.float32, device=GPU_DEVICE,
                )
                for i in range(0, len(voxels_gpu), chunk_size):
                    chunk = voxels_gpu[i:i+chunk_size]
                    idx   = shell_idx_gpu[i:i+chunk_size]
                    dists = torch.cdist(chunk, frame_atoms)
                    near  = (dists < protein_vdw_radius).any(dim=1)
                    accum[idx[near]] += 1
                del frame_atoms
            out = accum.cpu().numpy().reshape(nbins) / n_frames
            del accum
            torch.cuda.empty_cache()
            return out

        def _occ_cpu(atom_indices):
            pos_all    = trajectory.xyz[:, atom_indices, :] * 10.0
            accum      = np.zeros(nbins, dtype=np.float32)
            occ_flat   = accum.ravel()
            chunk_size = 5_000
            for fi in tqdm(range(n_frames)):
                fp = pos_all[fi]
                for i in range(0, len(shell_voxel_coords), chunk_size):
                    ch   = shell_voxel_coords[i:i+chunk_size]
                    ci   = shell_indices[i:i+chunk_size]
                    d    = np.linalg.norm(
                        ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                    near = (d < protein_vdw_radius).any(axis=1)
                    occ_flat[ci[near]] += 1
            return accum / n_frames

        _occ = _occ_gpu if USE_GPU else _occ_cpu

        # for each frame, per-residue-type: which shell voxels does it block?
        #  ════════════════════════════════════════════════════════════════
        #  GOAL: for every shell voxel, measure how often it is blocked
        #        by an aromatic ring — and whether that ring is actively
        #        contacting the ligand or just transiently passing through.
        #        (Contested-space values are a subset of this, taken by
        #        masking in Step 9 — the loop below never restricts to
        #        contested_mask directly.)
        #  ════════════════════════════════════════════════════════════════
        #
        #  THREE accumulators (all flat, shape: n_voxels):
        #
        #  occ_gpu   → voxel blocked by ANY aromatic ring centre
        #  con_gpu   → voxel blocked by a ring that IS   contacting ligand
        #  ncon_gpu  → voxel blocked by a ring that IS NOT contacting ligand
        #
        #  PER-FRAME INPUTS:
        #
        #  ring_centers_pos[fi]   ── shape: (n_aro_residues, 3)
        #  ┌────────────┬───┬───┬───┐
        #  │  Phe 42    │ x │ y │ z │  ← centroid of aromatic ring
        #  │  Tyr 87    │ x │ y │ z │
        #  │  Trp 113   │ x │ y │ z │
        #  └────────────┴───┴───┴───┘
        #
        #  aromatic_contacts[fi]  ── shape: (n_aro_residues,)  Bool
        #  ┌────────────┬────────┐
        #  │  Phe 42    │  True  │  ← ring is within contact distance of ligand
        #  │  Tyr 87    │  False │  ← ring is NOT contacting ligand this frame
        #  │  Trp 113   │  True  │
        #  └────────────┴────────┘
        #
        #  CHUNK LOOP LOGIC  (per chunk of shell voxels):
        #
        #  dists  ── (chunk_size, n_aro_residues)   cdist output
        #
        #              Phe42   Tyr87   Trp113
        #  voxel₀  [ [ 0.8  │  3.9  │  5.2 ]   near = dists < protein_vdw_radius
        #  voxel₁    [ 4.1  │  1.1  │  2.3 ]        ↓ .any(dim=1)
        #  voxel₂    [ 6.0  │  5.5  │  1.3 ] ]  ┌──────────────────┐
        #              │                          │  T  │  T  │  T  │  → occ_gpu
        #              │                          └──────────────────┘
        #              │
        #  near  ── (chunk_size, n_aro_residues)  Bool
        #
        #              Phe42   Tyr87  Trp113        frame_flags
        #  voxel₀  [ [  T   │   F  │   F  ]  AND  [  T  │  F  │  T  ]
        #  voxel₁    [  F   │   T  │   F  ]       [  T  │  F  │  T  ]
        #  voxel₂    [  F   │   F  │   T  ] ]     [  T  │  F  │  T  ]
        #              ↓ .any(dim=1)                   ↓ .any(dim=1)
        #         ┌──────────────────┐           ┌──────────────────┐
        #         │  T  │  F  │  T  │           │  F  │  F  │  T  │
        #         └──────────────────┘           └──────────────────┘
        #               con_gpu                       ncon_gpu
        #               (ring nearby                  (ring nearby BUT
        #                AND in contact)               not in contact)
        #
        #  AFTER ALL FRAMES  (divide by n_frames → occupancy fraction 0→1):
        #
        #  aromatic_occ              ── how often ANY ring blocks the voxel
        #  aromatic_contacts_occ     ── how often a BINDING ring blocks it
        #                               high → residue is a stable π-contact
        #                                      with the ligand → likely to
        #                                      compete for this space
        #  aromatic_non_contacts_occ ── transient ring passages only
        #                               high → flexible loop / solvent
        #                                      exposed ring swinging in
        #
        # Full-shell maps produced here; contested-space maps (nonzero only
        # inside contested_mask) are derived from these afterward — see
        # _mask_to_contested() in Step 9.
        #
        #  SPATIAL VIEW  (2D cross-section, shell region)
        #
        #  ┌──────────────────────────────────────────────────────────────┐
        #  │                                                              │
        #  │     ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·          │
        #  │     ·  ·  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ·  ·  ·          │
        #  │     ·  ▒  ▒  C  C  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ·  ·          │
        #  │     ·  ▒  C  C [Phe42] C  ▒  ▒  ▒  ▒  ▒  ▒  ·  ·         │
        #  │     ·  ▒  C  C  C  C  ▒  ╔══════════╗  ▒  ·  ·  ·         │
        #  │     ·  ▒  ▒  C  C  ▒  ▒  ║          ║  ▒  ·  ·  ·         │
        #  │     ·  ▒  ▒  ▒  ▒  ▒  ▒  ║  ligand  ║  ▒  ·  ·  ·         │
        #  │     ·  ▒  ▒  ▒  ▒  ▒  ▒  ║          ║  ▒  ·  ·  ·         │
        #  │     ·  ▒  N  N  N  N  ▒  ╚══════════╝  ▒  ·  ·  ·         │
        #  │     ·  ▒  ▒  N  N  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ·  ·  ·          │
        #  │     ·  ·  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ▒  ·  ·  ·          │
        #  │     ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·          │
        #  │                                                              │
        #  └──────────────────────────────────────────────────────────────┘
        #
        #  ·  outside shell (not visited)
        #  ▒  shell voxel, no aromatic ring nearby this frame
        #  C  shell voxel, blocked by Phe42 AND Phe42 contacts ligand
        #     → increments occ_gpu + con_gpu
        #  N  shell voxel, blocked by a ring NOT contacting ligand
        #     → increments occ_gpu + ncon_gpu
        #
        #  TWO REPRESENTATIVE FRAMES:
        #
        #  frame A  (Phe42 rotated toward ligand — in contact)
        #
        #       [Phe42]──────►  ╔════╗
        #        ring centre    ║lig.║     voxels between ring and ligand:
        #                       ╚════╝     near = T, contact = T  → C
        #
        #  frame B  (Phe42 swung away — NOT in contact)
        #
        #       [Phe42]
        #        ring                      voxels near ring:
        #           ↘  (away from ligand)  near = T, contact = F  → N
        #              ╔════╗
        #              ║lig.║
        #              ╚════╝
        #
        #  After N frames, the ratio C/(C+N) per voxel — restricted to
        #  contested_mask downstream — tells you:
        #  → high C  : ring is stably docked against ligand here
        #              → structural π-contact, hard to displace
        #  → high N  : ring only passes through transiently
        #              → flexible residue, voxel recoverable for ligand growth
        # ── Split occupancy helper: returns (total, contact, non_contact) ──────
        # Used for H-bond donors and acceptors.  contact_flags_2d is
        # (n_frames, n_protein_residues); atom_to_res maps each atom to its
        # residue column so we can look up the per-frame H-bond flag per atom.
        def _occ_split_gpu(atom_indices, atom_to_res, contact_flags_2d):
            accum_total = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            accum_con   = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            accum_ncon  = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            n_atoms    = len(atom_indices)
            chunk_size = max(1_000, min(200_000, int(4e9 / (n_atoms * 4))))
            for fi in tqdm(range(n_frames)):
                frame_atoms = torch.tensor(
                    trajectory.xyz[fi, atom_indices, :] * 10.0,
                    dtype=torch.float32, device=GPU_DEVICE,
                )
                frame_contact_mask = torch.tensor(
                    contact_flags_2d[fi, atom_to_res].astype(bool),
                    dtype=torch.bool, device=GPU_DEVICE,
                )
                for i in range(0, len(voxels_gpu), chunk_size):
                    chunk = voxels_gpu[i:i+chunk_size]
                    idx   = shell_idx_gpu[i:i+chunk_size]
                    dists = torch.cdist(chunk, frame_atoms)        # (chunk, n_atoms)
                    near  = dists < protein_vdw_radius             # (chunk, n_atoms)
                    accum_total[idx[near.any(dim=1)]] += 1
                    accum_con[idx[
                        torch.logical_and(near, frame_contact_mask.unsqueeze(0)).any(dim=1)
                    ]] += 1
                    accum_ncon[idx[
                        torch.logical_and(near, ~frame_contact_mask.unsqueeze(0)).any(dim=1)
                    ]] += 1
                del frame_atoms, frame_contact_mask
            total = accum_total.cpu().numpy().reshape(nbins) / n_frames
            con   = accum_con.cpu().numpy().reshape(nbins) / n_frames
            ncon  = accum_ncon.cpu().numpy().reshape(nbins) / n_frames
            del accum_total, accum_con, accum_ncon
            torch.cuda.empty_cache()
            return total, con, ncon

        def _occ_split_cpu(atom_indices, atom_to_res, contact_flags_2d):
            pos_all    = trajectory.xyz[:, atom_indices, :] * 10.0
            accum_total = np.zeros(nbins, dtype=np.float32)
            accum_con   = np.zeros(nbins, dtype=np.float32)
            accum_ncon  = np.zeros(nbins, dtype=np.float32)
            flat_total  = accum_total.ravel()
            flat_con    = accum_con.ravel()
            flat_ncon   = accum_ncon.ravel()
            chunk_size = 5_000
            for fi in tqdm(range(n_frames)):
                fp           = pos_all[fi]
                contact_mask = contact_flags_2d[fi, atom_to_res].astype(bool)
                for i in range(0, len(shell_voxel_coords), chunk_size):
                    ch   = shell_voxel_coords[i:i+chunk_size]
                    ci   = shell_indices[i:i+chunk_size]
                    d    = np.linalg.norm(
                        ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                    near = d < protein_vdw_radius
                    flat_total[ci[near.any(axis=1)]] += 1
                    flat_con[ci[  (near &  contact_mask[np.newaxis, :]).any(axis=1)]] += 1
                    flat_ncon[ci[ (near & ~contact_mask[np.newaxis, :]).any(axis=1)]] += 1
            return accum_total / n_frames, accum_con / n_frames, accum_ncon / n_frames

        _occ_split = _occ_split_gpu if USE_GPU else _occ_split_cpu

        # ========================================================================
        # STEP 7: AROMATIC OCCUPANCY (ring centroids, own inline split)
        # ========================================================================
        # Same three-accumulator contact/non-contact logic as Step 6, but using
        # ring_centers_pos / aromatic_contacts (one centroid + one contact flag
        # per aromatic RESIDUE, already frame-aligned — no atom_to_res lookup
        # needed) instead of _occ_split, since aromatic contacts are tracked
        # per ring rather than per raw atom:
        #
        #  ring_centers_pos[fi]   ── (n_aro_residues, 3)   one centroid per ring
        #  aromatic_contacts[fi]  ── (n_aro_residues,) bool  ring in ligand contact?
        #
        #  dists ── cdist(voxel_chunk, ring_centers)   (chunk, n_aro)
        #  near  ── dists < protein_vdw_radius
        #
        #  occ_gpu  += voxels where near.any(ring axis)              → any ring nearby
        #  con_gpu  += voxels where (near AND frame_flags).any(...)  → nearby ring IS in contact
        #  ncon_gpu += voxels where (near AND ~frame_flags).any(...) → nearby ring NOT in contact
        #
        #  SPATIAL INTERPRETATION (unchanged from the legacy version):
        #  high con  : ring stably docked against the ligand here →
        #              structural π-contact, hard to displace
        #  high ncon : ring only swings through this voxel transiently →
        #              flexible residue, space recoverable for ligand growth
        # ── Aromatic: ring centers + contact split ─────────────────────────────
        if n_aro_residues > 0:
            print("Aromatic ring-center occupancy …")
            if USE_GPU:
                occ_gpu  = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
                con_gpu  = (torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
                            if have_aro_contacts else None)
                ncon_gpu = (torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
                            if have_aro_contacts else None)
                chunk_size = max(1_000, min(200_000, int(4e9 / (n_aro_residues * 4))))

                for fi in tqdm(range(n_frames)):
                    frame_aro = torch.tensor(
                        ring_centers_pos[fi], dtype=torch.float32, device=GPU_DEVICE)
                    if have_aro_contacts:
                        frame_flags = torch.tensor(
                            aromatic_contacts[fi], dtype=torch.bool, device=GPU_DEVICE)

                    for i in range(0, len(voxels_gpu), chunk_size):
                        chunk = voxels_gpu[i:i+chunk_size]
                        idx   = shell_idx_gpu[i:i+chunk_size]
                        dists = torch.cdist(chunk, frame_aro)         # (chunk, n_aro)
                        near  = dists < protein_vdw_radius             # (chunk, n_aro) bool
                        occ_gpu[idx[near.any(dim=1)]] += 1

                        if have_aro_contacts:
                            con_gpu[idx[
                                torch.logical_and(near, frame_flags.unsqueeze(0)).any(dim=1)
                            ]] += 1
                            ncon_gpu[idx[
                                torch.logical_and(near, ~frame_flags.unsqueeze(0)).any(dim=1)
                            ]] += 1

                    del frame_aro
                    if have_aro_contacts:
                        del frame_flags

                aromatic_occ = occ_gpu.cpu().numpy().reshape(nbins) / n_frames
                del occ_gpu
                if have_aro_contacts:
                    aromatic_contacts_occ     = con_gpu.cpu().numpy().reshape(nbins) / n_frames
                    aromatic_non_contacts_occ = ncon_gpu.cpu().numpy().reshape(nbins) / n_frames
                    del con_gpu, ncon_gpu
                torch.cuda.empty_cache()

            else:  # CPU path
                occ_flat  = aromatic_occ.ravel()
                con_flat  = aromatic_contacts_occ.ravel()     if have_aro_contacts else None
                ncon_flat = aromatic_non_contacts_occ.ravel() if have_aro_contacts else None
                chunk_size = 5_000
                for fi in tqdm(range(n_frames)):
                    fp    = ring_centers_pos[fi]               # (n_aro, 3)
                    flags = aromatic_contacts[fi] if have_aro_contacts else None
                    for i in range(0, len(shell_voxel_coords), chunk_size):
                        ch   = shell_voxel_coords[i:i+chunk_size]
                        ci   = shell_indices[i:i+chunk_size]
                        d    = np.linalg.norm(
                            ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                        near = d < protein_vdw_radius          # (chunk, n_aro)
                        occ_flat[ci[near.any(axis=1)]] += 1
                        if have_aro_contacts:
                            con_flat[ci[ (near & flags[np.newaxis, :]).any(axis=1)]] += 1
                            ncon_flat[ci[(near & ~flags[np.newaxis, :]).any(axis=1)]] += 1
                aromatic_occ /= n_frames
                if have_aro_contacts:
                    aromatic_contacts_occ     /= n_frames
                    aromatic_non_contacts_occ /= n_frames

        # ========================================================================
        # STEP 8: REMAINING FEATURE TYPES
        # ========================================================================
        # Hydrophobic/positive/negative always use the plain (no-split) kernel —
        # there's no notion of "hydrophobic contact" or "charge contact" tracked
        # upstream, only aromatic and H-bond have a contact/non-contact source.
        # H-bond donor/acceptor use the split kernel only when
        # compute_hbond_contacts() was actually run (have_hbond_contacts),
        # falling back to the plain kernel otherwise.
        # ── Non-aromatic features ──────────────────────────────────────────────
        if len(hydrophobic_atoms) > 0:
            print("Hydrophobic occupancy …")
            hydrophobic_occ = _occ(hydrophobic_atoms)

        if len(hbond_donor_atoms) > 0:
            print("H-bond donor occupancy …")
            if have_hbond_contacts:
                hbond_donors_occ, hbond_donors_contacts_occ, hbond_donors_non_contacts_occ = \
                    _occ_split(hbond_donor_atoms, donor_atom_to_res,
                               self.hbond_contact_frames_pd)
            else:
                hbond_donors_occ = _occ(hbond_donor_atoms)

        if len(hbond_acceptor_atoms) > 0:
            print("H-bond acceptor occupancy …")
            if have_hbond_contacts:
                hbond_acceptors_occ, hbond_acceptors_contacts_occ, hbond_acceptors_non_contacts_occ = \
                    _occ_split(hbond_acceptor_atoms, acceptor_atom_to_res,
                               self.hbond_contact_frames_ld)
            else:
                hbond_acceptors_occ = _occ(hbond_acceptor_atoms)

        if len(positive_atoms) > 0:
            print("Positive charge occupancy …")
            positive_occ = _occ(positive_atoms)

        if len(negative_atoms) > 0:
            print("Negative charge occupancy …")
            negative_occ = _occ(negative_atoms)

        # GPU cleanup — unconditional
        if USE_GPU:
            del voxels_gpu, shell_idx_gpu
            torch.cuda.empty_cache()

        # ========================================================================
        # STEP 9: DERIVE CONTESTED-SPACE MAPS FROM THE FULL-SHELL RESULT
        # ========================================================================
        # Zero out every voxel outside contested_mask rather than recomputing —
        # the full-shell arrays already contain the contested voxels' correct
        # values, since contested_mask ⊆ full_shell_mask.
        # ── Derive contested maps by masking full-shell results ────────────────
        def _mask_to_contested(arr):
            if arr is None:
                return None
            out = arr.copy()
            out[~contested_mask] = 0.0
            return out

        full_maps = {
            'aromatic_occupancy':                     aromatic_occ,
            'aromatic_occupancy_contacts':            aromatic_contacts_occ,
            'aromatic_occupancy_non_contacts':        aromatic_non_contacts_occ,
            'hydrophobic_occupancy':                  hydrophobic_occ,
            'hbond_donors_occupancy':                 hbond_donors_occ,
            'hbond_donors_occupancy_contacts':        hbond_donors_contacts_occ,
            'hbond_donors_occupancy_non_contacts':    hbond_donors_non_contacts_occ,
            'hbond_acceptors_occupancy':              hbond_acceptors_occ,
            'hbond_acceptors_occupancy_contacts':     hbond_acceptors_contacts_occ,
            'hbond_acceptors_occupancy_non_contacts': hbond_acceptors_non_contacts_occ,
            'positive_occupancy':                     positive_occ,
            'negative_occupancy':                     negative_occ,
            'grid_info':                              grid_info,
        }
        contested_maps = {k: _mask_to_contested(v) if k != 'grid_info' else v
                          for k, v in full_maps.items()}

        self.pharmacophore_maps_full      = full_maps
        self.pharmacophore_maps_contested = contested_maps
        return contested_maps, full_maps

    def compute_pharmacophore_maps_threaded(self, protein_vdw_radius: float = 1.5,
                                             n_workers: int = None):
        """Multithreaded-CPU sibling of compute_pharmacophore_maps() — same
        algorithm, same result, same GPU path when PyTorch+CUDA is available.
        Only the three CPU kernels (_occ_cpu, _occ_split_cpu, and the aromatic
        ring-center CPU branch) differ: frames are split into n_workers
        contiguous ranges, each processed by its own thread into private
        accumulator(s), summed together at the end (occupancy counts are
        additive before the final /n_frames average). See
        compute_pharmacophore_maps() for the full parameter docs and return
        value description — identical here.

        Parameters
        ----------
        n_workers : int, optional
            Number of CPU threads for the fallback path. Defaults to
            DEFAULT_THREAD_WORKERS. Ignored when GPU acceleration is active.
        """
        if self.negative_space_data is None:
            raise RuntimeError(
                "Call compute_negative_space() before compute_pharmacophore_maps_threaded()"
            )

        trajectory        = self.prot_lig_traj
        grid_info         = self.negative_space_data['grid_info']
        xmin              = grid_info['xmin']
        dx                = grid_info['dx']
        nbins             = grid_info['nbins']
        full_shell_mask   = self.negative_space_data['full_shell_mask']
        contested_mask    = self.negative_space_data['contested_space_mask']

        n_frames = trajectory.n_frames
        n_voxels = int(np.prod(nbins))
        if not USE_GPU:
            print(f"  CPU threads           : "
                  f"{max(1, min(n_workers or DEFAULT_THREAD_WORKERS, n_frames))}")

        # ── Atom selections + aromatic ring groups (identical) ─────────────────
        protein_rings, protein_rings_index, _ = self.get_protein_rings()

        n_aro_residues = len(protein_rings)
        ring_centers_pos = np.zeros((n_frames, n_aro_residues, 3), dtype=np.float32)
        for res_idx, atom_indices in enumerate(protein_rings):
            ring_centers_pos[:, res_idx, :] = (
                trajectory.xyz[:, atom_indices, :].mean(axis=1) * 10.0
            )

        have_aro_contacts = (self._stacked_full is not None
                             and len(protein_rings_index) > 0)
        if have_aro_contacts:
            aromatic_contacts = self._stacked_full[:, self._protein_rings_index].astype(bool)
        else:
            aromatic_contacts = None
            print("  compute_aromatic_contacts() not called — contact/non-contact "
                  "split will be skipped")

        have_hbond_contacts = (self.hbond_contact_frames_pd is not None and
                               self.hbond_contact_frames_ld is not None)
        if not have_hbond_contacts:
            print("  compute_hbond_contacts() not called — H-bond contact/non-contact "
                  "split will be skipped")

        hydrophobic_atoms = self.prot_lig_top.select(
            'protein and (resname ALA VAL LEU ILE MET PHE TRP PRO) '
            'and (name CB CG CG1 CG2 CD CD1 CD2 CE CE1 CE2 CZ)'
        )
        hbond_donor_atoms = np.union1d(
            self.prot_lig_top.select('protein and element N'),
            self.prot_lig_top.select(
                'protein and element O and resname SER THR TYR ASN GLN'
            ),
        )
        hbond_acceptor_atoms = self.prot_lig_top.select(
            'protein and (element O or element N)'
        )
        positive_atoms = self.prot_lig_top.select(
            'protein and resname LYS ARG HIS and name NZ NH1 NH2 NE2'
        )
        negative_atoms = self.prot_lig_top.select(
            'protein and resname ASP GLU and name OD1 OD2 OE1 OE2'
        )

        print(f"  Aromatic ring groups  : {n_aro_residues}")
        print(f"  Hydrophobic C atoms   : {len(hydrophobic_atoms)}")
        print(f"  H-bond donor atoms    : {len(hbond_donor_atoms)}")
        print(f"  H-bond acceptor atoms : {len(hbond_acceptor_atoms)}")
        print(f"  Positive charge atoms : {len(positive_atoms)}"
              + (" [none — score stays zero]" if len(positive_atoms) == 0 else ""))
        print(f"  Negative charge atoms : {len(negative_atoms)}"
              + (" [none — score stays zero]" if len(negative_atoms) == 0 else ""))

        donor_atom_to_res = np.array(
            [self.prot_lig_top.atom(a).residue.index for a in hbond_donor_atoms],
            dtype=int,
        )
        acceptor_atom_to_res = np.array(
            [self.prot_lig_top.atom(a).residue.index for a in hbond_acceptor_atoms],
            dtype=int,
        )

        # ── Voxel grid — full shell (identical) ─────────────────────────────────
        x, y, z      = (np.arange(nbins[i]) * dx + xmin[i] for i in range(3))
        xx, yy, zz   = np.meshgrid(x, y, z, indexing='ij')
        voxel_coords = np.stack([xx, yy, zz], axis=-1)

        shell_voxel_coords = voxel_coords[full_shell_mask]
        shell_indices      = np.where(full_shell_mask.ravel())[0]

        print(f"\nScoring {len(shell_voxel_coords):,} shell voxels "
              f"(full shell — contested derived by masking) …")

        aromatic_occ      = np.zeros(nbins, dtype=np.float32)
        aromatic_contacts_occ     = (np.zeros(nbins, dtype=np.float32)
                                     if have_aro_contacts else None)
        aromatic_non_contacts_occ = (np.zeros(nbins, dtype=np.float32)
                                     if have_aro_contacts else None)
        hydrophobic_occ   = np.zeros(nbins, dtype=np.float32)
        hbond_donors_occ              = np.zeros(nbins, dtype=np.float32)
        hbond_donors_contacts_occ     = (np.zeros(nbins, dtype=np.float32)
                                         if have_hbond_contacts else None)
        hbond_donors_non_contacts_occ = (np.zeros(nbins, dtype=np.float32)
                                         if have_hbond_contacts else None)
        hbond_acceptors_occ              = np.zeros(nbins, dtype=np.float32)
        hbond_acceptors_contacts_occ     = (np.zeros(nbins, dtype=np.float32)
                                            if have_hbond_contacts else None)
        hbond_acceptors_non_contacts_occ = (np.zeros(nbins, dtype=np.float32)
                                            if have_hbond_contacts else None)
        positive_occ      = np.zeros(nbins, dtype=np.float32)
        negative_occ      = np.zeros(nbins, dtype=np.float32)

        if USE_GPU:
            voxels_gpu     = torch.tensor(
                shell_voxel_coords, dtype=torch.float32, device=GPU_DEVICE)
            shell_idx_gpu  = torch.tensor(
                shell_indices, dtype=torch.long, device=GPU_DEVICE)

        # ── Binary occupancy kernel: non-aromatic features ──────────────────────
        def _occ_gpu(atom_indices):
            accum      = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            n_atoms    = len(atom_indices)
            chunk_size = max(1_000, min(200_000, int(4e9 / (n_atoms * 4))))
            for fi in tqdm(range(n_frames)):
                frame_atoms = torch.tensor(
                    trajectory.xyz[fi, atom_indices, :] * 10.0,
                    dtype=torch.float32, device=GPU_DEVICE,
                )
                for i in range(0, len(voxels_gpu), chunk_size):
                    chunk = voxels_gpu[i:i+chunk_size]
                    idx   = shell_idx_gpu[i:i+chunk_size]
                    dists = torch.cdist(chunk, frame_atoms)
                    near  = (dists < protein_vdw_radius).any(dim=1)
                    accum[idx[near]] += 1
                del frame_atoms
            out = accum.cpu().numpy().reshape(nbins) / n_frames
            del accum
            torch.cuda.empty_cache()
            return out

        def _occ_cpu_threaded(atom_indices):
            # Multithreaded version of compute_pharmacophore_maps()'s _occ_cpu.
            pos_all    = trajectory.xyz[:, atom_indices, :] * 10.0
            chunk_size = 5_000

            def _partial(start, stop):
                local = np.zeros(nbins, dtype=np.float32)
                local_flat = local.ravel()
                for fi in range(start, stop):
                    fp = pos_all[fi]
                    for i in range(0, len(shell_voxel_coords), chunk_size):
                        ch   = shell_voxel_coords[i:i+chunk_size]
                        ci   = shell_indices[i:i+chunk_size]
                        d    = np.linalg.norm(
                            ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                        near = (d < protein_vdw_radius).any(axis=1)
                        local_flat[ci[near]] += 1
                return local

            accum = _threaded_frame_reduce(n_frames, _partial, _sum_reduce, n_workers=n_workers)
            return accum / n_frames

        _occ = _occ_gpu if USE_GPU else _occ_cpu_threaded

        # ── Split occupancy helper: returns (total, contact, non_contact) ──────
        def _occ_split_gpu(atom_indices, atom_to_res, contact_flags_2d):
            accum_total = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            accum_con   = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            accum_ncon  = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
            n_atoms    = len(atom_indices)
            chunk_size = max(1_000, min(200_000, int(4e9 / (n_atoms * 4))))
            for fi in tqdm(range(n_frames)):
                frame_atoms = torch.tensor(
                    trajectory.xyz[fi, atom_indices, :] * 10.0,
                    dtype=torch.float32, device=GPU_DEVICE,
                )
                frame_contact_mask = torch.tensor(
                    contact_flags_2d[fi, atom_to_res].astype(bool),
                    dtype=torch.bool, device=GPU_DEVICE,
                )
                for i in range(0, len(voxels_gpu), chunk_size):
                    chunk = voxels_gpu[i:i+chunk_size]
                    idx   = shell_idx_gpu[i:i+chunk_size]
                    dists = torch.cdist(chunk, frame_atoms)        # (chunk, n_atoms)
                    near  = dists < protein_vdw_radius             # (chunk, n_atoms)
                    accum_total[idx[near.any(dim=1)]] += 1
                    accum_con[idx[
                        torch.logical_and(near, frame_contact_mask.unsqueeze(0)).any(dim=1)
                    ]] += 1
                    accum_ncon[idx[
                        torch.logical_and(near, ~frame_contact_mask.unsqueeze(0)).any(dim=1)
                    ]] += 1
                del frame_atoms, frame_contact_mask
            total = accum_total.cpu().numpy().reshape(nbins) / n_frames
            con   = accum_con.cpu().numpy().reshape(nbins) / n_frames
            ncon  = accum_ncon.cpu().numpy().reshape(nbins) / n_frames
            del accum_total, accum_con, accum_ncon
            torch.cuda.empty_cache()
            return total, con, ncon

        def _occ_split_cpu_threaded(atom_indices, atom_to_res, contact_flags_2d):
            # Multithreaded version of compute_pharmacophore_maps()'s
            # _occ_split_cpu: each thread returns its own (total, con, ncon)
            # triple for its frame range; summed elementwise at the end.
            pos_all    = trajectory.xyz[:, atom_indices, :] * 10.0
            chunk_size = 5_000

            def _partial(start, stop):
                local_total = np.zeros(nbins, dtype=np.float32)
                local_con   = np.zeros(nbins, dtype=np.float32)
                local_ncon  = np.zeros(nbins, dtype=np.float32)
                flat_total  = local_total.ravel()
                flat_con    = local_con.ravel()
                flat_ncon   = local_ncon.ravel()
                for fi in range(start, stop):
                    fp           = pos_all[fi]
                    contact_mask = contact_flags_2d[fi, atom_to_res].astype(bool)
                    for i in range(0, len(shell_voxel_coords), chunk_size):
                        ch   = shell_voxel_coords[i:i+chunk_size]
                        ci   = shell_indices[i:i+chunk_size]
                        d    = np.linalg.norm(
                            ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                        near = d < protein_vdw_radius
                        flat_total[ci[near.any(axis=1)]] += 1
                        flat_con[ci[  (near &  contact_mask[np.newaxis, :]).any(axis=1)]] += 1
                        flat_ncon[ci[ (near & ~contact_mask[np.newaxis, :]).any(axis=1)]] += 1
                return local_total, local_con, local_ncon

            total, con, ncon = _threaded_frame_reduce(
                n_frames, _partial, _tuple_sum_reduce, n_workers=n_workers)
            return total / n_frames, con / n_frames, ncon / n_frames

        _occ_split = _occ_split_gpu if USE_GPU else _occ_split_cpu_threaded

        # ── Aromatic occupancy (ring centroids, own inline split) ──────────────
        if n_aro_residues > 0:
            print("Aromatic ring-center occupancy …")
            if USE_GPU:
                occ_gpu  = torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
                con_gpu  = (torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
                            if have_aro_contacts else None)
                ncon_gpu = (torch.zeros(n_voxels, dtype=torch.float32, device=GPU_DEVICE)
                            if have_aro_contacts else None)
                chunk_size = max(1_000, min(200_000, int(4e9 / (n_aro_residues * 4))))

                for fi in tqdm(range(n_frames)):
                    frame_aro = torch.tensor(
                        ring_centers_pos[fi], dtype=torch.float32, device=GPU_DEVICE)
                    if have_aro_contacts:
                        frame_flags = torch.tensor(
                            aromatic_contacts[fi], dtype=torch.bool, device=GPU_DEVICE)

                    for i in range(0, len(voxels_gpu), chunk_size):
                        chunk = voxels_gpu[i:i+chunk_size]
                        idx   = shell_idx_gpu[i:i+chunk_size]
                        dists = torch.cdist(chunk, frame_aro)         # (chunk, n_aro)
                        near  = dists < protein_vdw_radius             # (chunk, n_aro) bool
                        occ_gpu[idx[near.any(dim=1)]] += 1

                        if have_aro_contacts:
                            con_gpu[idx[
                                torch.logical_and(near, frame_flags.unsqueeze(0)).any(dim=1)
                            ]] += 1
                            ncon_gpu[idx[
                                torch.logical_and(near, ~frame_flags.unsqueeze(0)).any(dim=1)
                            ]] += 1

                    del frame_aro
                    if have_aro_contacts:
                        del frame_flags

                aromatic_occ = occ_gpu.cpu().numpy().reshape(nbins) / n_frames
                del occ_gpu
                if have_aro_contacts:
                    aromatic_contacts_occ     = con_gpu.cpu().numpy().reshape(nbins) / n_frames
                    aromatic_non_contacts_occ = ncon_gpu.cpu().numpy().reshape(nbins) / n_frames
                    del con_gpu, ncon_gpu
                torch.cuda.empty_cache()

            else:  # Multithreaded CPU path
                # Same per-frame math as compute_pharmacophore_maps()'s aromatic
                # CPU branch; each thread gets its own (occ, con, ncon) triple
                # for its frame range, summed elementwise at the end.
                chunk_size = 5_000

                def _partial_aromatic(start, stop):
                    local_occ  = np.zeros(nbins, dtype=np.float32)
                    local_con  = np.zeros(nbins, dtype=np.float32) if have_aro_contacts else None
                    local_ncon = np.zeros(nbins, dtype=np.float32) if have_aro_contacts else None
                    occ_flat  = local_occ.ravel()
                    con_flat  = local_con.ravel()  if have_aro_contacts else None
                    ncon_flat = local_ncon.ravel() if have_aro_contacts else None
                    for fi in range(start, stop):
                        fp    = ring_centers_pos[fi]               # (n_aro, 3)
                        flags = aromatic_contacts[fi] if have_aro_contacts else None
                        for i in range(0, len(shell_voxel_coords), chunk_size):
                            ch   = shell_voxel_coords[i:i+chunk_size]
                            ci   = shell_indices[i:i+chunk_size]
                            d    = np.linalg.norm(
                                ch[:, np.newaxis, :] - fp[np.newaxis, :, :], axis=2)
                            near = d < protein_vdw_radius          # (chunk, n_aro)
                            occ_flat[ci[near.any(axis=1)]] += 1
                            if have_aro_contacts:
                                con_flat[ci[ (near & flags[np.newaxis, :]).any(axis=1)]] += 1
                                ncon_flat[ci[(near & ~flags[np.newaxis, :]).any(axis=1)]] += 1
                    return local_occ, local_con, local_ncon

                def _combine_aromatic(parts):
                    occs  = [p[0] for p in parts]
                    occ   = reduce(np.add, occs)
                    if have_aro_contacts:
                        con  = reduce(np.add, [p[1] for p in parts])
                        ncon = reduce(np.add, [p[2] for p in parts])
                    else:
                        con  = None
                        ncon = None
                    return occ, con, ncon

                aromatic_occ, aromatic_contacts_occ, aromatic_non_contacts_occ = \
                    _threaded_frame_reduce(n_frames, _partial_aromatic, _combine_aromatic,
                                           n_workers=n_workers)
                aromatic_occ /= n_frames
                if have_aro_contacts:
                    aromatic_contacts_occ     /= n_frames
                    aromatic_non_contacts_occ /= n_frames

        # ── Remaining feature types (identical dispatch) ───────────────────────
        if len(hydrophobic_atoms) > 0:
            print("Hydrophobic occupancy …")
            hydrophobic_occ = _occ(hydrophobic_atoms)

        if len(hbond_donor_atoms) > 0:
            print("H-bond donor occupancy …")
            if have_hbond_contacts:
                hbond_donors_occ, hbond_donors_contacts_occ, hbond_donors_non_contacts_occ = \
                    _occ_split(hbond_donor_atoms, donor_atom_to_res,
                               self.hbond_contact_frames_pd)
            else:
                hbond_donors_occ = _occ(hbond_donor_atoms)

        if len(hbond_acceptor_atoms) > 0:
            print("H-bond acceptor occupancy …")
            if have_hbond_contacts:
                hbond_acceptors_occ, hbond_acceptors_contacts_occ, hbond_acceptors_non_contacts_occ = \
                    _occ_split(hbond_acceptor_atoms, acceptor_atom_to_res,
                               self.hbond_contact_frames_ld)
            else:
                hbond_acceptors_occ = _occ(hbond_acceptor_atoms)

        if len(positive_atoms) > 0:
            print("Positive charge occupancy …")
            positive_occ = _occ(positive_atoms)

        if len(negative_atoms) > 0:
            print("Negative charge occupancy …")
            negative_occ = _occ(negative_atoms)

        if USE_GPU:
            del voxels_gpu, shell_idx_gpu
            torch.cuda.empty_cache()

        def _mask_to_contested(arr):
            if arr is None:
                return None
            out = arr.copy()
            out[~contested_mask] = 0.0
            return out

        full_maps = {
            'aromatic_occupancy':                     aromatic_occ,
            'aromatic_occupancy_contacts':            aromatic_contacts_occ,
            'aromatic_occupancy_non_contacts':        aromatic_non_contacts_occ,
            'hydrophobic_occupancy':                  hydrophobic_occ,
            'hbond_donors_occupancy':                 hbond_donors_occ,
            'hbond_donors_occupancy_contacts':        hbond_donors_contacts_occ,
            'hbond_donors_occupancy_non_contacts':    hbond_donors_non_contacts_occ,
            'hbond_acceptors_occupancy':              hbond_acceptors_occ,
            'hbond_acceptors_occupancy_contacts':     hbond_acceptors_contacts_occ,
            'hbond_acceptors_occupancy_non_contacts': hbond_acceptors_non_contacts_occ,
            'positive_occupancy':                     positive_occ,
            'negative_occupancy':                     negative_occ,
            'grid_info':                              grid_info,
        }
        contested_maps = {k: _mask_to_contested(v) if k != 'grid_info' else v
                          for k, v in full_maps.items()}

        self.pharmacophore_maps_full      = full_maps
        self.pharmacophore_maps_contested = contested_maps
        return contested_maps, full_maps

    def define_pharmacophore(self):
        """Compute residue-type pharmacophore maps on contested and full-shell spaces.

        Thin orchestration wrapper around compute_pharmacophore_maps().
        Requires compute_negative_space() first; compute_aromatic_contacts()
        is optional but enables the aromatic contact/non-contact split.

        Returns
        -------
        (maps_contested, maps_full) — same as compute_pharmacophore_maps().
        """
        if self.negative_space_data is None:
            raise RuntimeError("Call compute_negative_space() before define_pharmacophore()")
        return self.compute_pharmacophore_maps()

    # ── 6. Output ─────────────────────────────────────────────────────────────

    def save_mrc_files(self, output_dir: str,
                       gaussian_sigma: Optional[float] = None,
                       score_percentile_tiers: Optional[List[int]] = None) -> Dict:
        """Export all computed maps as MRC volume files for PyMOL visualization.

        Writes MRC files organised into subdirectories:
          negative_space/          space masks + free fraction + ligand distance
          growth_features/         chemical feature score maps for growth space
                                   (if compute_growth_space_features() was called)
          pharmacophore_contested/ residue-type occupancy on contested space
          pharmacophore_full/      residue-type occupancy on full shell
                                   (if define_pharmacophore() was called)

        Also writes ligand_centroid.pdb (most representative ligand frame,
        requires scikit-learn).

        Requires compute_negative_space() to have been called first.

        Parameters
        ----------
        output_dir : str
            Root directory for all output files.
        gaussian_sigma : float, optional
            Gaussian smoothing applied to all maps before writing (in voxels).
            None = no smoothing.  Recommended range: 0.5–1.5.
        score_percentile_tiers : list of int, optional
            For each tier N in this list, an additional filtered growth-feature
            map is written retaining only the top-N% highest-scoring voxels.
            Default: [1, 5, 10, 20].

        Returns
        -------
        dict mapping descriptive key → absolute file path for every file written.
        """
        if not MRCFILE_AVAILABLE:
            raise ImportError("mrcfile required — install: pip install mrcfile")
        if self.negative_space_data is None:
            raise RuntimeError(
                "Call compute_negative_space() before save_mrc_files()")

        if score_percentile_tiers is None:
            score_percentile_tiers = [1, 5, 10, 20]

        os.makedirs(output_dir, exist_ok=True)
        mrc_files: Dict = {}

        grid_info = self.negative_space_data['grid_info']
        xmin = grid_info['xmin']
        dx   = grid_info['dx']

        # ── Ligand centroid PDB ───────────────────────────────────────────────
        if SKLEARN_AVAILABLE and self.ligand_traj is not None:
            print("Computing ligand centroid PDB …")
            lig_xyz     = self.ligand_traj.xyz.reshape(self.ligand_traj.n_frames, -1)
            dists_mat   = _sklearn_pairwise_distances(lig_xyz)
            centroid_fi = int(dists_mat.sum(axis=1).argmin())
            ligand_pdb  = os.path.join(output_dir, 'ligand_centroid.pdb')
            self.ligand_traj[centroid_fi].save_pdb(ligand_pdb)
            mrc_files['ligand_centroid_pdb'] = ligand_pdb
            print(f"  ✓ ligand_centroid.pdb")
        elif not SKLEARN_AVAILABLE:
            print("  Warning: scikit-learn not available — ligand centroid PDB skipped")

        # ── Section A: Negative space maps ────────────────────────────────────
        ns_dir = os.path.join(output_dir, 'negative_space')
        os.makedirs(ns_dir, exist_ok=True)
        nd = self.negative_space_data
        print("\nExporting negative space MRC files …")

        for key, filename in [
            ('growth_space_mask',    'growth_space.mrc'),
            ('contested_space_mask', 'contested_space.mrc'),
            ('occupied_space_mask',  'occupied_space.mrc'),
            ('free_fraction',        'free_fraction.mrc'),
        ]:
            fname = os.path.join(ns_dir, filename)
            _write_mrc_field(nd[key].astype(np.float32), xmin, dx, fname, gaussian_sigma)
            mrc_files[key] = fname
            print(f"  ✓ {filename}")

        # Distance to ligand (inf → 0 for MRC compatibility)
        dist_field = np.where(nd['min_dist_to_ligand'] < np.inf,
                              nd['min_dist_to_ligand'], 0.0).astype(np.float32)
        fname = os.path.join(ns_dir, 'distance_to_ligand.mrc')
        _write_mrc_field(dist_field, xmin, dx, fname, gaussian_sigma)
        mrc_files['distance_to_ligand'] = fname
        print(f"  ✓ distance_to_ligand.mrc")

        # Combined categorical: 0=outside shell, 1=growth, 2=contested, 3=occupied
        combined = np.zeros(nd['growth_space_mask'].shape, dtype=np.float32)
        combined[nd['growth_space_mask']]    = 1.0
        combined[nd['contested_space_mask']] = 2.0
        combined[nd['occupied_space_mask']]  = 3.0
        fname = os.path.join(ns_dir, 'combined_space.mrc')
        _write_mrc_field(combined, xmin, dx, fname, gaussian_sigma)
        mrc_files['combined_space'] = fname
        print(f"  ✓ combined_space.mrc")

        # ── Section B: Growth space feature maps ──────────────────────────────
        if self.growth_space_features is not None:
            gf_dir = os.path.join(output_dir, 'growth_features')
            os.makedirs(gf_dir, exist_ok=True)
            print("\nExporting growth feature MRC files …")

            feature_files = [
                ('aromatic_score',        'aromatic_sites.mrc'),
                ('hydrophobic_score',     'hydrophobic_sites.mrc'),
                ('hbond_donor_score',     'hbond_donor_sites.mrc'),
                ('hbond_acceptor_score',  'hbond_acceptor_sites.mrc'),
                ('positive_charge_score', 'positive_charge_sites.mrc'),
                ('negative_charge_score', 'negative_charge_sites.mrc'),
            ]
            for key, filename in feature_files:
                score_arr = self.growth_space_features[key]

                # Full map (all nonzero voxels)
                fname = os.path.join(gf_dir, filename)
                _write_mrc_field(score_arr, xmin, dx, fname)
                mrc_files[f'growth_{key}'] = fname
                print(f"  ✓ {filename}")

                # Percentile-tiered maps: top-N% of nonzero voxels only
                nonzero = score_arr[score_arr > 0]
                if len(nonzero) == 0:
                    print(f"    (skipping tiers for {key}: no nonzero voxels)")
                    continue
                for tier in score_percentile_tiers:
                    threshold     = np.percentile(nonzero, 100 - tier)
                    filtered      = np.where(score_arr >= threshold,
                                             score_arr, 0.0).astype(np.float32)
                    base, ext     = os.path.splitext(filename)
                    tier_filename = f'{base}_top{tier}pct{ext}'
                    tier_fname    = os.path.join(gf_dir, tier_filename)
                    _write_mrc_field(filtered, xmin, dx, tier_fname)
                    mrc_files[f'growth_{key}_top{tier}pct'] = tier_fname
                    n_sur = int(np.count_nonzero(filtered))
                    print(f"    ✓ {tier_filename} "
                          f"({n_sur}/{len(nonzero)} voxels, "
                          f"threshold={threshold:.3f})")

        # ── Section C: Pharmacophore occupancy maps ────────────────────────────
        pharm_targets = [
            ('pharmacophore_contested', self.pharmacophore_maps_contested),
            ('pharmacophore_full',      self.pharmacophore_maps_full),
        ]
        occ_keys = [
            'aromatic_occupancy',
            'aromatic_occupancy_contacts',
            'aromatic_occupancy_non_contacts',
            'hydrophobic_occupancy',
            'hbond_donors_occupancy',
            'hbond_donors_occupancy_contacts',
            'hbond_donors_occupancy_non_contacts',
            'hbond_acceptors_occupancy',
            'hbond_acceptors_occupancy_contacts',
            'hbond_acceptors_occupancy_non_contacts',
            'positive_occupancy',
            'negative_occupancy',
        ]
        for subdir_name, maps_dict in pharm_targets:
            if maps_dict is None:
                continue
            pm_dir = os.path.join(output_dir, subdir_name)
            os.makedirs(pm_dir, exist_ok=True)
            print(f"\nExporting {subdir_name} MRC files …")
            add_tiers = (subdir_name == 'pharmacophore_contested')
            for key in occ_keys:
                field = maps_dict.get(key)
                if field is None:
                    print(f"  Skipping {key} (not computed)")
                    continue
                fname = os.path.join(pm_dir, f'{key}.mrc')
                _write_mrc_field(field.astype(np.float32), xmin, dx,
                                 fname, gaussian_sigma)
                mrc_files[f'{subdir_name}/{key}'] = fname
                print(f"  ✓ {key}.mrc")

                if add_tiers:
                    nonzero = field[field > 0]
                    if len(nonzero) == 0:
                        print(f"    (skipping tiers for {key}: no nonzero voxels)")
                        continue
                    for tier in score_percentile_tiers:
                        threshold  = np.percentile(nonzero, 100 - tier)
                        filtered   = np.where(field >= threshold,
                                              field, 0.0).astype(np.float32)
                        tier_fname = os.path.join(pm_dir, f'{key}_top{tier}pct.mrc')
                        _write_mrc_field(filtered, xmin, dx, tier_fname, gaussian_sigma)
                        mrc_files[f'{subdir_name}/{key}_top{tier}pct'] = tier_fname
                        n_sur = int(np.count_nonzero(filtered))
                        print(f"    ✓ {key}_top{tier}pct.mrc "
                              f"({n_sur}/{len(nonzero)} voxels, "
                              f"threshold={threshold:.3f})")

        return mrc_files

    def compute_residue_dot_positions(self,
                                      mask_key: str = 'full_shell_mask',
                                      stride: int = 1) -> Dict:
        """Collect per-frame protein atom positions for PyMOL dot visualization.

        Uses the same atom selections as compute_pharmacophore_maps():

          aromatic    : ring centroids (TYR/PHE/HIS/TRP) per frame per ring,
                        split into contact / non-contact using _stacked_full
          hydrophobic : sidechain carbon centroid per residue per frame (no split)
          hba         : protein O atom positions per frame (H-bond acceptors),
                        split via hbond_contact_frames_ld (ligand-donor frames)
          hbd         : H atoms bonded to N or O per frame (H-bond donors),
                        split via hbond_contact_frames_pd (protein-donor frames)

        Positions are in **absolute Å** — the same coordinate system used by
        the MRC voxel files.  No ligand-centroid subtraction is applied.
        (plot_3d_residues() subtracts the centroid for Plotly display only.)

        Contact/non-contact splits fall back gracefully: if the required
        compute_aromatic_contacts() / compute_hbond_contacts() result is not
        available, all points are placed in the non-contact bucket.

        Points are filtered to positions that fall inside the voxel mask
        defined by ``mask_key``.

        Parameters
        ----------
        mask_key : str
            Voxel mask used as spatial filter.
            One of 'full_shell_mask', 'growth_space_mask',
            'contested_space_mask'.
        stride : int
            Use every nth frame (1 = all frames).  Larger values reduce the
            number of dots saved to the PDB files.

        Stores
        ------
        residue_dot_positions : dict with keys:
            'aromatic_contact'     → np.ndarray (N, 3) Å
            'aromatic_non_contact' → np.ndarray (N, 3) Å
            'hydrophobic'          → np.ndarray (N, 3) Å
            'hba_contact'          → np.ndarray (N, 3) Å
            'hba_non_contact'      → np.ndarray (N, 3) Å
            'hbd_contact'          → np.ndarray (N, 3) Å
            'hbd_non_contact'      → np.ndarray (N, 3) Å

        Returns
        -------
        residue_dot_positions dict (same object as self.residue_dot_positions)
        """
        if self.negative_space_data is None:
            raise RuntimeError("Call compute_negative_space() first")
        if self.prot_lig_traj is None:
            raise RuntimeError("Call load() first")
        if mask_key not in self.negative_space_data:
            raise ValueError(
                f"mask_key '{mask_key}' not found. "
                f"Valid: {[k for k in self.negative_space_data if 'mask' in k]}"
            )

        import time as _time
        t0 = _time.time()

        grid_info = self.negative_space_data['grid_info']
        xmin  = grid_info['xmin']
        dx    = grid_info['dx']
        nbins = np.array(grid_info['nbins'])
        mask  = self.negative_space_data[mask_key]   # (nx, ny, nz) bool

        traj = self.prot_lig_traj[::stride]

        def _in_shell(pos_aa):
            """pos_aa : (N, 3) absolute Å → bool (N,)."""
            idx = np.round((pos_aa - xmin) / dx).astype(int)
            idx = np.clip(idx, 0, nbins - 1)
            fi  = idx[:, 0] * nbins[1] * nbins[2] + idx[:, 1] * nbins[2] + idx[:, 2]
            return mask.ravel()[fi]

        def _stack(lst):
            return np.vstack(lst) if lst else np.empty((0, 3), dtype=np.float32)

        # ── Aromatic: ring centroid per frame, contact/non-contact split ──
        protein_rings, protein_rings_index, _ = self.get_protein_rings()
        have_aro = (self._stacked_full is not None and len(protein_rings_index) > 0)
        if not have_aro and len(protein_rings_index) > 0:
            print("  compute_aromatic_contacts() not called — all aromatic dots "
                  "placed in non-contact bucket")

        aro_con_xyz, aro_ncon_xyz = [], []
        for ring_atoms, res_idx in zip(protein_rings, protein_rings_index):
            pos = traj.xyz[:, ring_atoms, :].mean(axis=1) * 10.0   # (F, 3) Å
            sel = _in_shell(pos)
            pos_sel = pos[sel]
            if len(pos_sel) == 0:
                continue
            if have_aro:
                flags = self._stacked_full[::stride][sel, res_idx].astype(bool)
                if flags.any():
                    aro_con_xyz.append(pos_sel[flags])
                if (~flags).any():
                    aro_ncon_xyz.append(pos_sel[~flags])
            else:
                aro_ncon_xyz.append(pos_sel)

        # ── Hydrophobic: sidechain carbon centroid per frame (no split) ───
        hph_xyz = []
        if self.hydrophobic_residue_atoms_dict:
            for res_label, atom_idx in self.hydrophobic_residue_atoms_dict.items():
                if len(atom_idx) == 0:
                    continue
                pos = traj.xyz[:, atom_idx, :].mean(axis=1) * 10.0
                sel = _in_shell(pos)
                if sel.any():
                    hph_xyz.append(pos[sel])

        # ── H-bond acceptors: protein O atoms, contact split via ld matrix ─
        acc_idx = self.prot_lig_top.select('protein and element O')
        have_hba = (self.hbond_contact_frames_ld is not None)
        if not have_hba and len(acc_idx) > 0:
            print("  compute_hbond_contacts() not called — all HBA dots "
                  "placed in non-contact bucket")

        hba_con_xyz, hba_ncon_xyz = [], []
        for atom_i in acc_idx:
            pos = traj.xyz[:, atom_i, :] * 10.0   # (F, 3)
            sel = _in_shell(pos)
            pos_sel = pos[sel]
            if len(pos_sel) == 0:
                continue
            if have_hba:
                r_idx = self.prot_lig_top.atom(atom_i).residue.index
                flags = self.hbond_contact_frames_ld[::stride][sel, r_idx].astype(bool)
                if flags.any():
                    hba_con_xyz.append(pos_sel[flags])
                if (~flags).any():
                    hba_ncon_xyz.append(pos_sel[~flags])
            else:
                hba_ncon_xyz.append(pos_sel)

        # ── H-bond donors: H on N/O, contact split via pd matrix ──────────
        hbd_indices = []
        for a0, a1 in self.prot_lig_top.bonds:
            pair_elems = {a0.element.symbol, a1.element.symbol}
            if pair_elems not in ({'N', 'H'}, {'O', 'H'}):
                continue
            if not (a0.residue.is_protein and a1.residue.is_protein):
                continue
            h_at = a1 if a1.element.symbol == 'H' else a0
            hbd_indices.append(h_at.index)

        have_hbd = (self.hbond_contact_frames_pd is not None)
        if not have_hbd and len(hbd_indices) > 0:
            print("  compute_hbond_contacts() not called — all HBD dots "
                  "placed in non-contact bucket")

        hbd_con_xyz, hbd_ncon_xyz = [], []
        for atom_i in hbd_indices:
            pos = traj.xyz[:, atom_i, :] * 10.0   # (F, 3)
            sel = _in_shell(pos)
            pos_sel = pos[sel]
            if len(pos_sel) == 0:
                continue
            if have_hbd:
                r_idx = self.prot_lig_top.atom(atom_i).residue.index
                flags = self.hbond_contact_frames_pd[::stride][sel, r_idx].astype(bool)
                if flags.any():
                    hbd_con_xyz.append(pos_sel[flags])
                if (~flags).any():
                    hbd_ncon_xyz.append(pos_sel[~flags])
            else:
                hbd_ncon_xyz.append(pos_sel)

        self.residue_dot_positions = {
            'aromatic_contact':     _stack(aro_con_xyz),
            'aromatic_non_contact': _stack(aro_ncon_xyz),
            'hydrophobic':          _stack(hph_xyz),
            'hba_contact':          _stack(hba_con_xyz),
            'hba_non_contact':      _stack(hba_ncon_xyz),
            'hbd_contact':          _stack(hbd_con_xyz),
            'hbd_non_contact':      _stack(hbd_ncon_xyz),
        }

        n_total = sum(len(v) for v in self.residue_dot_positions.values())
        print(f"compute_residue_dot_positions: {n_total:,} total points "
              f"(mask={mask_key}, stride={stride})")
        for cat, arr in self.residue_dot_positions.items():
            print(f"  {cat:22s}: {len(arr):,}")
        print(f"\tDone in {round(_time.time() - t0, 2)} s")
        return self.residue_dot_positions

    def save_aromatic_ring_trajectory(self, output_dir: str, stride: int = 1) -> Dict[str, str]:
        """Write a multi-model PDB of aromatic ring centroids for time-resolved PyMOL visualization.

        One MODEL block per frame (stride-subsampled). Each model contains:
        - Chain A: one pseudo-atom per protein aromatic ring (TYR/PHE/HIS/TRP)
        - Chain B: one pseudo-atom per ligand aromatic ring

        B-factor encodes per-frame contact status:
        - Protein rings: 1.0 if aromatic stacking contact in that frame, 0.0 otherwise
        - Ligand rings: 0.5 (constant landmark)

        The companion .pml colors spheres blue→red by B-factor and scales radius by contact
        strength, so scrubbing through PyMOL states shows contact timing.

        Requires compute_aromatic_contacts() for B-factor coloring; degrades to all-zero
        B-factor if not called.

        Parameters
        ----------
        output_dir : str
            Root output directory.
        stride : int
            Frame subsampling interval (default 1 = every frame).

        Returns
        -------
        dict with keys 'pdb' and 'pml' mapping to absolute file paths.
        """
        if self.prot_lig_traj is None:
            raise RuntimeError("Call load() first")

        os.makedirs(output_dir, exist_ok=True)

        protein_rings, protein_rings_index, _ = self.get_protein_rings()
        lig_rings = [r for r in self.get_ligand_rings() if r['aromatic']]

        # Build ligand ring atom index arrays in prot_lig context
        lig_ring_atom_idx = []
        for ring in lig_rings:
            idx_list = []
            for name in ring['atom_names']:
                idxs = self.convert_ligand_atom_name(name)
                if len(idxs) > 0:
                    idx_list.append(idxs[0])
            lig_ring_atom_idx.append(np.array(idx_list, dtype=int))

        traj     = self.prot_lig_traj[::stride]
        n_frames = traj.n_frames

        if self._stacked_full is not None:
            stacked = self._stacked_full[::stride]
        else:
            print("Warning: _stacked_full not set — call compute_aromatic_contacts() first. "
                  "All protein ring B-factors will be 0.0.")
            stacked = None

        pdb_path = os.path.abspath(os.path.join(f'{output_dir}/dots', 'aromatic_ring_centers.pdb'))
        pml_path = os.path.abspath(os.path.join(output_dir, 'view_aromatic_rings.pml'))

        with open(pdb_path, 'w') as f:
            for fi in range(n_frames):
                f.write(f"MODEL     {fi + 1:4d}\n")
                serial = 1

                # Chain A: protein aromatic ring centroids
                for ring_atoms, res_idx in zip(protein_rings, protein_rings_index):
                    xyz_nm   = traj.xyz[fi, ring_atoms, :]
                    centroid = xyz_nm.mean(axis=0) * 10.0   # nm → Å
                    bfac     = float(stacked[fi, res_idx]) if stacked is not None else 0.0
                    res      = self.prot_lig_top.residue(res_idx)
                    resname  = res.name[:3]
                    resseq   = res.resSeq + self.offset
                    f.write(
                        f"ATOM  {serial:5d}  CA  {resname:<3s} A{resseq:4d}    "
                        f"{centroid[0]:8.3f}{centroid[1]:8.3f}{centroid[2]:8.3f}"
                        f"  1.00{bfac:6.2f}\n"
                    )
                    serial += 1

                # Chain B: ligand aromatic ring centroids
                for li, atom_idx in enumerate(lig_ring_atom_idx):
                    if len(atom_idx) == 0:
                        continue
                    xyz_nm   = traj.xyz[fi, atom_idx, :]
                    centroid = xyz_nm.mean(axis=0) * 10.0
                    f.write(
                        f"ATOM  {serial:5d}  CA  LIG B{li + 1:4d}    "
                        f"{centroid[0]:8.3f}{centroid[1]:8.3f}{centroid[2]:8.3f}"
                        f"  1.00  0.50\n"
                    )
                    serial += 1

                f.write("ENDMDL\n")
            f.write("END\n")

        with open(pml_path, 'w') as f:
            f.write(f"load {pdb_path}, ring_centers\n")
            f.write("show spheres, ring_centers\n")
            f.write("set sphere_mode, 1\n")
            # VDW is a global atom property (not per-state) — fixed sizes by chain
            f.write("alter ring_centers and chain A, vdw=0.4\n")
            f.write("alter ring_centers and chain B, vdw=0.6\n")
            f.write("rebuild\n")
            f.write("bg_color white\n")
            f.write("\n")
            # mdo commands fire on every frame change (scrubbing or playback).
            # At frame s, ring_centers shows state s, so 'b > 0.6' evaluates
            # against that state's B-factors — this is the correct per-frame
            # coloring mechanism in PyMOL (cmd.color has no 'state' parameter).
            f.write("python\n")
            f.write("from pymol import cmd\n")
            f.write("n = cmd.count_states('ring_centers')\n")
            f.write("cmd.mclear()\n")
            f.write("cmd.do('mset 1 -' + str(n))\n")
            f.write("_recolor = (\n")
            f.write("    'color gray70, ring_centers ;'\n")
            f.write("    'color red,    ring_centers and b > 0.6 ;'\n")
            f.write("    'color blue,   ring_centers and b < 0.1 ;'\n")
            f.write("    'color gold,   ring_centers and b > 0.3 and b < 0.6'\n")
            f.write(")\n")
            f.write("for s in range(1, n + 1):\n")
            f.write("    cmd.mdo(s, _recolor)\n")
            f.write("print(f'ring_centers: {n} frames. '\n")
            f.write("      'Use the Frame slider or mplay to navigate.')\n")
            f.write("python end\n")
            f.write("\n")
            f.write("# Static initial coloring for frame 1\n")
            f.write("color gray70, ring_centers\n")
            f.write("color red,    ring_centers and b > 0.6\n")
            f.write("color blue,   ring_centers and b < 0.1\n")
            f.write("color gold,   ring_centers and b > 0.3 and b < 0.6\n")

        n_prot = len(protein_rings)
        n_lig  = len(lig_rings)
        print(f"save_aromatic_ring_trajectory: {n_frames} frames (stride={stride}), "
              f"{n_prot} protein ring(s), {n_lig} ligand ring(s)")
        print(f"  PDB: {pdb_path}")
        print(f"  PML: {pml_path}")

        return {'pdb': pdb_path, 'pml': pml_path}

    def save_dots_pdb(self, output_dir: str) -> Dict[str, str]:
        """Write per-category dot-cloud PDB files for PyMOL visualization.

        Each contact/non-contact split is written as a **separate PDB file**
        so they become independent PyMOL objects that can be toggled on/off
        individually (``enable dots_aromatic_contact`` etc.).

        Creates ``{output_dir}/dots/`` and writes up to 7 PDB files:
          aromatic_contact.pdb     (residue ARO — frames with stacking contact)
          aromatic_non_contact.pdb (residue ARO — frames without stacking contact)
          hydrophobic.pdb          (residue HPH — all frames, no split)
          hba_contact.pdb          (residue HBA — frames with H-bond acceptor contact)
          hba_non_contact.pdb      (residue HBA — frames without H-bond acceptor contact)
          hbd_contact.pdb          (residue HBD — frames with H-bond donor contact)
          hbd_non_contact.pdb      (residue HBD — frames without H-bond donor contact)

        Files are skipped silently if the corresponding array is empty.

        Requires compute_residue_dot_positions() to have been called first.

        Parameters
        ----------
        output_dir : str
            Root output directory (same as used for save_mrc_files()).

        Returns
        -------
        dict mapping file stem → absolute path of written PDB file.
        """
        if self.residue_dot_positions is None:
            raise RuntimeError("Call compute_residue_dot_positions() first")

        dots_dir = os.path.join(output_dir, 'dots')
        os.makedirs(dots_dir, exist_ok=True)

        rdp = self.residue_dot_positions
        elem_C = md.element.Element.getBySymbol('C')

        def _write_pdb(coords_aa, res_name, fname):
            """Write a single-category PDB (all C atoms); return fname or None."""
            n = len(coords_aa)
            if n == 0:
                return None
            top   = md.Topology()
            chain = top.add_chain()
            res   = top.add_residue(res_name, chain)
            for _ in range(n):
                top.add_atom('C', elem_C, res)
            xyz_nm = (coords_aa / 10.0).reshape(1, -1, 3)
            md.Trajectory(xyz_nm, top).save_pdb(fname)
            return fname

        written = {}

        _categories = [
            ('aromatic_contact',     'ARO', 'aromatic_contact.pdb'),
            ('aromatic_non_contact', 'ARO', 'aromatic_non_contact.pdb'),
            ('hydrophobic',          'HPH', 'hydrophobic.pdb'),
            ('hba_contact',          'HBA', 'hba_contact.pdb'),
            ('hba_non_contact',      'HBA', 'hba_non_contact.pdb'),
            ('hbd_contact',          'HBD', 'hbd_contact.pdb'),
            ('hbd_non_contact',      'HBD', 'hbd_non_contact.pdb'),
        ]
        for key, res_name, pdb_name in _categories:
            coords = rdp.get(key, np.empty((0, 3)))
            fname  = os.path.join(dots_dir, pdb_name)
            result = _write_pdb(coords, res_name, fname)
            if result:
                written[key] = result
                print(f"  {pdb_name:<28}: {len(coords):,} atoms")

        print(f"save_dots_pdb: wrote {len(written)} file(s) to {dots_dir}")
        return written

    def write_pymol_script(self, output_dir: str,
                           score_percentile_tiers: Optional[List[int]] = None,
                           set_view: Optional[str] = None,
                           load_trajectory: bool = False) -> str:
        """Write a PyMOL .pml script loading all MRC volumes from save_mrc_files().

        Reconstructs file paths from the same deterministic directory structure
        written by save_mrc_files(). Missing files are silently skipped so the
        script works regardless of which analysis methods were called.

        Loads:
          - ligand_centroid.pdb as sticks
          - growth_space.mrc and contested_space.mrc as volumetric renders
          - Pharmacophore occupancy maps (contested) as isomesh at level 0.3
          - Growth-feature score maps as isomesh at level 0.3 (full maps shown;
            percentile-tiered maps loaded but hidden — enable with 'enable <name>')

        Requires save_mrc_files() to have been called with the same output_dir.

        Parameters
        ----------
        output_dir : str
            Same root directory passed to save_mrc_files().
        score_percentile_tiers : list of int, optional
            Tiers to load (default: [1, 5, 10, 20]).  Must match what was used
            in save_mrc_files() to find the tiered .mrc files.
        set_view : str, optional
            Camera orientation as a raw string copied directly from PyMOL's
            ``get_view`` command output. Use a raw Python string to preserve
            the backslashes, e.g.::

                r\"\"\"set_view (\\
                    -0.055,  -0.422,   0.904,\\
                    ...
                    82.718, 127.118, -20.000 )\"\"\"

            When provided the active ``set_view`` command is written to the
            script. When None (default) a commented-out template is written.
        load_trajectory : bool, optional
            When True, prepend ``load`` commands for the topology and trajectory
            files that were used to construct this object (self._topology_path and
            self._trajectory_path). Default False.

        Returns
        -------
        str — absolute path of the written .pml script.
        """
        if score_percentile_tiers is None:
            score_percentile_tiers = [1, 5, 10, 20]

        # Reconstruct expected subdirectory paths
        ns_dir = os.path.join(output_dir, 'negative_space')
        gf_dir = os.path.join(output_dir, 'growth_features')
        pc_dir = os.path.join(output_dir, 'pharmacophore_contested')

        ligand_pdb       = os.path.join(output_dir, 'ligand_centroid.pdb')
        growth_space_mrc = os.path.join(ns_dir, 'growth_space.mrc')
        contested_mrc    = os.path.join(ns_dir, 'contested_space.mrc')

        pymol_script = os.path.join(output_dir, 'view_pharmacophore.pml')
        with open(pymol_script, 'w') as f:
            f.write("# PyMOL pharmacophore visualization script\n")
            f.write(f"# Generated by PharmacophoreTrajectory.write_pymol_script()\n\n")

            # ── Trajectory ───────────────────────────────────────────────────
            if load_trajectory:
                f.write("# MD trajectory\n")
                f.write(f"load {self._topology_path}\n")
                f.write(f"load_traj {self._trajectory_path}, {os.path.splitext(os.path.basename(self._topology_path))[0]}\n\n")
                f.write(f"hide spheres, {os.path.splitext(os.path.basename(self._topology_path))[0]}\n")
                f.write(f"disable {os.path.splitext(os.path.basename(self._topology_path))[0]}\n")
                f.write("show sticks, resn PHE+TYR+TRP+HIS\n")
                f.write("hide cartoon\n")

            # ── Ligand ────────────────────────────────────────────────────────
            if os.path.exists(ligand_pdb):
                f.write("# Ligand centroid structure\n")
                f.write(f"load {ligand_pdb}, ligand\n")
                f.write("show sticks, ligand\n")
                f.write("util.cbag ligand\n")
                f.write("set stick_radius, 0.15\n\n")

                # ── SASA Jet coloring (overrides element colors if computed) ──
                if self.sasa_ligand_atoms_df is not None:
                    f.write(
                        "# Ligand SASA coloring (Jet: darkblue=buried → red=exposed)\n"
                        "# Computed on protein+ligand trajectory — no explicit solvent:\n"
                        "# low SASA = atom embedded in protein pocket;\n"
                        "# high SASA = atom pointing outward / solvent-accessible.\n"
                    )
                    mean_sasa = self.sasa_ligand_atoms_df.mean(axis=0)
                    smin, smax = float(mean_sasa.min()), float(mean_sasa.max())
                    span = smax - smin if smax > smin else 1.0
                    f.write(
                        f"# SASA range: {smin*100:.2f}–{smax*100:.2f} Å²\n"
                    )
                    for atom_name, sasa_val in mean_sasa.items():
                        t = (float(sasa_val) - smin) / span
                        hex_col = _jet_hex(t)
                        f.write(
                            f"color 0x{hex_col}, ligand and name {atom_name}\n"
                        )
                    f.write("\n")

            # ── Growth space volume ────────────────────────────────────────────
            if os.path.exists(growth_space_mrc):
                f.write("# Growth space (protein occupancy < 30%)\n")
                f.write(f"load {growth_space_mrc}, growth_space\n")
                f.write("volume growth_vol, growth_space\n")
                f.write("cmd.volume_ramp_new('ramp816', [\\\n")
                f.write("     -0.21, 0.33, 1.00, 1.00, 0.02, \\\n")
                f.write("      0.07, 0.00, 0.00, 1.00, 0.02, \\\n")
                f.write("    ])\n")
                f.write("volume_color growth_vol, ramp816\n\n")

            # ── Contested space volume ─────────────────────────────────────────
            if os.path.exists(contested_mrc):
                f.write("# Contested space (protein occupancy >= 30%)\n")
                f.write(f"load {contested_mrc}, contested_space\n")
                f.write("volume contested_vol, contested_space\n")
                f.write("cmd.volume_ramp_new('ramp505', [\\\n")
                f.write("      0.10, 1.00, 0.47, 0.00, 0.18, \\\n")
                f.write("      0.61, 1.00, 0.55, 0.04, 0.05, \\\n")
                f.write("    ])\n")
                f.write("volume_color contested_vol, ramp505\n\n")

            # ── Pharmacophore maps (contested space) ──────────────────────────
            f.write("# PHARMACOPHORE FROM CONTESTED SPACE\n")
            contested_occ = [
                ('aromatic_occupancy',                     'black'),
                ('aromatic_occupancy_contacts',            'purpleblue'),
                ('aromatic_occupancy_non_contacts',        'deepsalmon'),
                ('hydrophobic_occupancy',                  'forest'),
                ('hbond_donors_occupancy',                 'firebrick'),
                ('hbond_donors_occupancy_contacts',        'purpleblue'),
                ('hbond_donors_occupancy_non_contacts',    'deepsalmon'),
                ('hbond_acceptors_occupancy',              'firebrick'),
                ('hbond_acceptors_occupancy_contacts',     'purpleblue'),
                ('hbond_acceptors_occupancy_non_contacts', 'deepsalmon'),
            ]
            for key, color in contested_occ:
                mrc_path = os.path.join(pc_dir, f'{key}.mrc')
                if not os.path.exists(mrc_path):
                    continue
                f.write(f"load {mrc_path}, {key}\n")
                f.write(f"isomesh {key}_mesh, {key}, 0.3\n")
                f.write(f"color {color}, {key}_mesh\n")
                for tier in score_percentile_tiers:
                    tier_mrc = os.path.join(pc_dir, f'{key}_top{tier}pct.mrc')
                    if os.path.exists(tier_mrc):
                        tier_obj = f'{key}_top{tier}pct'
                        f.write(f"load {tier_mrc}, {tier_obj}\n")
                        f.write(f"isomesh {tier_obj}_mesh, {tier_obj}, 0.01\n")
                        f.write(f"color {color}, {tier_obj}_mesh\n")
                        f.write(f"disable {tier_obj}_mesh\n")
            f.write("\n")

            # ── Growth space chemical feature maps ────────────────────────────
            f.write("# CHEMICAL FEATURES (GROWTH SPACE)\n")
            f.write("# Full maps shown; tiered maps loaded but hidden.\n")
            f.write("# Toggle tiers: enable <name>_mesh / disable <name>_mesh\n\n")

            feature_vis = [
                ('aromatic_sites',        'aromatic',   'black',    'Aromatic sites'),
                ('hydrophobic_sites',     'hydrophobic','forest',   'Hydrophobic sites'),
                ('hbond_acceptor_sites',  'hbond_acc',  'firebrick','H-bond acceptor sites'),
                ('hbond_donor_sites',     'hbond_don',  'firebrick','H-bond donor sites'),
                ('negative_charge_sites', 'neg_charge', 'purple',   'Negative charge sites'),
                ('positive_charge_sites', 'pos_charge', 'magenta',  'Positive charge sites'),
            ]
            for fname_base, obj_name, color, comment in feature_vis:
                full_mrc = os.path.join(gf_dir, f'{fname_base}.mrc')
                if not os.path.exists(full_mrc):
                    continue
                f.write(f"# {comment}\n")
                f.write(f"load {full_mrc}, {obj_name}\n")
                f.write(f"isomesh {obj_name}_mesh, {obj_name}, 0.3\n")
                f.write(f"color {color}, {obj_name}_mesh\n")
                for tier in score_percentile_tiers:
                    tier_mrc = os.path.join(gf_dir, f'{fname_base}_top{tier}pct.mrc')
                    if os.path.exists(tier_mrc):
                        tier_obj = f'{obj_name}_top{tier}pct'
                        f.write(f"load {tier_mrc}, {tier_obj}\n")
                        f.write(f"isomesh {tier_obj}_mesh, {tier_obj}, 0.01\n")
                        f.write(f"color {color}, {tier_obj}_mesh\n")
                        f.write(f"disable {tier_obj}_mesh\n")
                f.write("\n")

            # ── Residue dot-cloud PDB files (save_dots_pdb output) ───────────
            # Each contact/non-contact split is a separate PyMOL object so it
            # can be toggled independently with enable/disable.
            # All objects are disabled by default; toggle with e.g.:
            #   enable dots_aromatic_contact
            #   enable dots_aromatic_non_contact
            dots_dir = os.path.join(output_dir, 'dots')
            _dot_cfg = [
                ('aromatic_contact',     'aromatic_contact.pdb',      'magenta', 0.10),
                ('aromatic_non_contact', 'aromatic_non_contact.pdb',  'grey60',  0.07),
                ('hydrophobic',          'hydrophobic.pdb',           'orange',  0.07),
                ('hba_contact',          'hba_contact.pdb',           'forest',  0.07),
                ('hba_non_contact',      'hba_non_contact.pdb',       'grey60',  0.05),
                ('hbd_contact',          'hbd_contact.pdb',           'blue',    0.07),
                ('hbd_non_contact',      'hbd_non_contact.pdb',       'grey60',  0.05),
            ]
            any_dots = False
            for stem, pdb_file, color, sphere_scale in _dot_cfg:
                pdb_path = os.path.join(dots_dir, pdb_file)
                if not os.path.exists(pdb_path):
                    continue
                if not any_dots:
                    f.write("# Per-frame residue atom positions (PDB dot clouds)\n")
                    f.write("# Each object is independently togglable:\n")
                    f.write("#   enable dots_aromatic_contact / dots_aromatic_non_contact\n")
                    f.write("#   enable dots_hba_contact / dots_hba_non_contact  etc.\n\n")
                    any_dots = True
                obj = f'dots_{stem}'
                f.write(f"load {pdb_path}, {obj}\n")
                f.write(f"hide licorice, {obj}\n")
                f.write(f"show spheres, {obj}\n")
                f.write(f"set sphere_scale, {sphere_scale}, {obj}\n")
                f.write(f"color {color}, {obj}\n")
                f.write(f"disable {obj}\n\n")

            # ── Aromatic ring centroid trajectory ────────────────────────────
            ring_centers_pdb = os.path.join(output_dir, 'aromatic_ring_centers.pdb')
            if os.path.exists(ring_centers_pdb):
                f.write(f"load {ring_centers_pdb}, ring_centers\n")
                f.write("show nb_spheres, ring_centers\n")
                f.write("color yellow, ring_centers\n")

            f.write('hide cartoon\n\n')
            # ── View settings ─────────────────────────────────────────────────
            f.write('disable contested_vol\n')
            f.write('disable aromatic_occupancy_mesh\n')
            f.write('disable aromatic_occupancy_contacts_mesh\n')
            f.write('disable aromatic_occupancy_non_contacts_mesh\n')
            f.write('disable hydrophobic_occupancy_mesh\n')
            f.write('disable hbond_donors_occupancy_mesh\n')
            f.write('disable hbond_acceptors_occupancy_mesh\n')
            f.write('disable aromatic_mesh\n')
            f.write('disable hydrophobic_mesh\n')
            f.write('disable hbond_acc_mesh\n')
            f.write('disable hbond_don_mesh\n')
            f.write('disable hbond_donors_occupancy_contacts_mesh\n')
            f.write('disable hbond_donors_occupancy_non_contacts_mesh\n')
            f.write('disable hbond_acceptors_occupancy_contacts_mesh\n')
            f.write('disable hbond_acceptors_occupancy_non_contacts_mesh\n')
            f.write('disable neg_charge_mesh\n')
            f.write('disable pos_charge_mesh\n')
            f.write('disable ring_centers\n\n')

            f.write("# View settings\n")
            f.write("bg_color white\n")
            f.write("set depth_cue, 0\n")
            if os.path.exists(ligand_pdb):
                f.write("center ligand\n")
            f.write("refresh\n\n")
            if os.path.exists(growth_space_mrc):
                f.write("zoom growth_vol\n")
            if set_view is not None and set_view.strip():
                f.write(set_view.strip() + "\n\n")
            else:
                f.write("# Uncomment and adjust for a custom saved view:\n")
                f.write("# set_view (\\\n")
                f.write("#     -0.055175416,   -0.422412157,    0.904722989,\\\n")
                f.write("#      0.977541029,   -0.207427859,   -0.037231173,\\\n")
                f.write("#      0.203391120,    0.882350326,    0.424371362,\\\n")
                f.write("#      0.000000000,    0.000000000, -104.918716431,\\\n")
                f.write("#     27.569999695,  -10.580001831,   42.870002747,\\\n")
                f.write("#     82.718719482,  127.118713379,  -20.000000000 )\n\n")
            f.write("set ray_trace_mode, 3\n")
            f.write("set ray_volume, 1\n")
            
            # ── Production images ───────────────────────────────────────────────
            f.write("enable contested_vol\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore.png, 1090, 1090\n")
            f.write("disable contested_vol\n")
            
            f.write("enable aromatic_occupancy_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_aromatics.png, 1090, 1090\n")
            f.write("disable aromatic_occupancy_top10pct_mesh\n")
            
            f.write("enable aromatic_occupancy_contacts_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_aromatics_contacts.png, 1090, 1090\n")
            f.write("disable aromatic_occupancy_contacts_top10pct_mesh\n")
            
            f.write("enable aromatic_occupancy_non_contacts_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_aromatics_non_contacts.png, 1090, 1090\n")
            f.write("disable aromatic_occupancy_non_contacts_top10pct_mesh\n")
            
            f.write("enable hydrophobic_occupancy_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hydrophobic.png, 1090, 1090\n")
            f.write("disable hydrophobic_occupancy_top10pct_mesh\n")
            
            f.write("enable hbond_donors_occupancy_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hbd.png, 1090, 1090\n")
            f.write("disable hbond_donors_occupancy_top10pct_mesh\n")
            
            f.write("enable hbond_donors_occupancy_contacts_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hbd_contacts.png, 1090, 1090\n")
            f.write("disable hbond_donors_occupancy_contacts_top10pct_mesh\n")
            
            f.write("enable hbond_donors_occupancy_non_contacts_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hbd_non_contacts.png, 1090, 1090\n")
            f.write("disable hbond_donors_occupancy_non_contacts_top10pct_mesh\n")
            
            f.write("enable hbond_acceptors_occupancy_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hba.png, 1090, 1090\n")
            f.write("disable hbond_acceptors_occupancy_top10pct_mesh\n")
            
            f.write("enable hbond_acceptors_occupancy_contacts_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hba_contacts.png, 1090, 1090\n")
            f.write("disable hbond_acceptors_occupancy_contacts_top10pct_mesh\n")
            
            f.write("enable hbond_acceptors_occupancy_non_contacts_top10pct_mesh\n")
            f.write("refresh\n")
            f.write(f"png {output_dir}/pharmacophore_hba_non_contacts.png, 1090, 1090\n")
            f.write("disable hbond_acceptors_occupancy_non_contacts_top10pct_mesh\n")
            
            f.write("# disable contested_vol\n")
            
            f.write("# quit\n")

        print(f"✓ PyMOL script: {pymol_script}")
        print(f"  Open with: pymol {pymol_script}")
        return pymol_script

    def plot_3d_residues(self, mask_key: str = 'full_shell_mask',
                         stride: int = 1,
                         marker_size: int = 1,
                         title: str = '') -> 'go.Figure':
        """3D scatter of per-frame residue atom positions around the ligand.

        One point per (trajectory frame, residue representative atom), filtered
        to positions that fall inside the voxel region defined by mask_key.
        The mean ligand centroid (over all frames) is placed at the origin.

        Categories
        ----------
        Ligand       : mean heavy-atom positions — one point per atom (static)
        Aromatic     : ring centroid per frame  (TYR / PHE / HIS / TRP)
        Hydrophobic  : sidechain C/S centroid per residue per frame
        HB acceptor  : protein O atom positions per frame
        HB donor     : protein H atoms bonded to N or O, per frame

        Parameters
        ----------
        mask_key : str
            Key in negative_space_data selecting the voxel region to display.
            Valid values: 'full_shell_mask', 'growth_space_mask',
            'contested_space_mask'.
        stride : int
            Use every nth frame (1 = all frames).
        marker_size : int
            Marker radius for residue dots.
        title : str
            Figure title.  Auto-generated when empty.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if go is None:
            raise ImportError("plotly required — install: pip install plotly")
        if self.negative_space_data is None:
            raise RuntimeError("Call compute_negative_space() first")
        if self.prot_lig_traj is None:
            raise RuntimeError("Call load() first")
        if mask_key not in self.negative_space_data:
            raise ValueError(
                f"mask_key '{mask_key}' not in negative_space_data. "
                f"Valid: {[k for k in self.negative_space_data if 'mask' in k]}")

        grid_info = self.negative_space_data['grid_info']
        xmin  = grid_info['xmin']
        dx    = grid_info['dx']
        nbins = np.array(grid_info['nbins'])
        mask  = self.negative_space_data[mask_key]     # (nx, ny, nz) bool

        traj     = self.prot_lig_traj[::stride]
        n_frames = traj.n_frames

        # ── Shell membership test (vectorised) ───────────────────────────────
        def _in_shell(pos_aa):
            """pos_aa : (N, 3) absolute Å → bool (N,)."""
            idx = np.round((pos_aa - xmin) / dx).astype(int)
            idx = np.clip(idx, 0, nbins - 1)
            fi  = idx[:, 0] * nbins[1] * nbins[2] + idx[:, 1] * nbins[2] + idx[:, 2]
            return mask.ravel()[fi]

        # ── Centering: mean ligand heavy-atom centroid over all frames ────────
        lig_aa  = traj.xyz[:, self.all_ligand_atoms_noh, :] * 10.0  # (F, L, 3)
        centroid = lig_aa.mean(axis=(0, 1))                          # (3,)  Å

        # ── Ligand trace: mean atom positions (one point per heavy atom) ──────
        lig_mean = lig_aa.mean(axis=0) - centroid                    # (L, 3)
        lig_names = [self.prot_lig_top.atom(int(i)).name
                     for i in self.all_ligand_atoms_noh]

        # ── Aromatic: ring centroid per frame ─────────────────────────────────
        protein_rings, protein_rings_index, _ = self.get_protein_rings()
        aro_xyz, aro_text = [], []
        for ring_atoms, res_idx in zip(protein_rings, protein_rings_index):
            res   = self.prot_lig_top.residue(res_idx)
            label = f"{res.name} {res.resSeq + self.offset}"
            pos   = traj.xyz[:, ring_atoms, :].mean(axis=1) * 10.0  # (F, 3)
            sel   = _in_shell(pos)
            if sel.any():
                aro_xyz.append(pos[sel] - centroid)
                aro_text.extend([label] * int(sel.sum()))

        # ── Hydrophobic: per-residue sidechain C/S centroid per frame ─────────
        hph_xyz, hph_text = [], []
        if self.hydrophobic_residue_atoms_dict:
            for res_label, atom_idx in self.hydrophobic_residue_atoms_dict.items():
                if len(atom_idx) == 0:
                    continue
                pos = traj.xyz[:, atom_idx, :].mean(axis=1) * 10.0  # (F, 3)
                sel = _in_shell(pos)
                if sel.any():
                    hph_xyz.append(pos[sel] - centroid)
                    hph_text.extend(
                        [res_label.replace('_', ' ')] * int(sel.sum()))

        # ── H-bond acceptors: protein O atom positions per frame ──────────────
        acc_idx = self.prot_lig_top.select('protein and element O')
        hba_xyz, hba_text = [], []
        for atom_i in acc_idx:
            at  = self.prot_lig_top.atom(int(atom_i))
            lbl = (f"{at.name} "
                   f"{at.residue.name} {at.residue.resSeq + self.offset}")
            pos = traj.xyz[:, atom_i, :] * 10.0   # (F, 3)
            sel = _in_shell(pos)
            if sel.any():
                hba_xyz.append(pos[sel] - centroid)
                hba_text.extend([lbl] * int(sel.sum()))

        # ── H-bond donors: H atoms bonded to N or O, per frame ───────────────
        # MDTraj bonds live on the Topology object, not on individual atoms.
        donor_pairs = []   # list of (h_atom_index, label)
        for a0, a1 in self.prot_lig_top.bonds:
            pair_elems = {a0.element.symbol, a1.element.symbol}
            if pair_elems not in ({'N', 'H'}, {'O', 'H'}):
                continue
            if not (a0.residue.is_protein and a1.residue.is_protein):
                continue
            h_at     = a1 if a1.element.symbol == 'H' else a0
            heavy_at = a0 if a0.element.symbol != 'H' else a1
            lbl = (f"{h_at.name}→{heavy_at.name} "
                   f"{h_at.residue.name} {h_at.residue.resSeq + self.offset}")
            donor_pairs.append((h_at.index, lbl))

        hbd_xyz, hbd_text = [], []
        for atom_i, lbl in donor_pairs:
            pos = traj.xyz[:, atom_i, :] * 10.0   # (F, 3)
            sel = _in_shell(pos)
            if sel.any():
                hbd_xyz.append(pos[sel] - centroid)
                hbd_text.extend([lbl] * int(sel.sum()))

        # ── Stack lists ───────────────────────────────────────────────────────
        def _stack(lst):
            return np.vstack(lst) if lst else np.empty((0, 3))

        aro_xyz = _stack(aro_xyz)
        hph_xyz = _stack(hph_xyz)
        hba_xyz = _stack(hba_xyz)
        hbd_xyz = _stack(hbd_xyz)

        # ── Build figure ──────────────────────────────────────────────────────
        fig = go.Figure()

        def _add_trace(xyz, text, name, color):
            if len(xyz) == 0:
                return
            fig.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode='markers',
                name=name,
                marker=dict(size=marker_size, color=color, opacity=0.2,
                            line=dict(width=0)),
                text=text,
                hovertemplate='%{text}<extra>' + name + '</extra>',
            ))

        fig.add_trace(go.Scatter3d(
            x=lig_mean[:, 0], y=lig_mean[:, 1], z=lig_mean[:, 2],
            mode='markers',
            name='Ligand',
            marker=dict(size=5, color='white',
                        line=dict(color='grey', width=1)),
            text=lig_names,
            hovertemplate='%{text}<extra>Ligand</extra>',
        ))
        _add_trace(aro_xyz, aro_text, 'Aromatic',    '#e377c2')
        _add_trace(hph_xyz, hph_text, 'Hydrophobic', '#ff7f0e')
        _add_trace(hba_xyz, hba_text, 'HB acceptor', '#2ca02c')
        _add_trace(hbd_xyz, hbd_text, 'HB donor',    '#1f77b4')

        n_total = sum(len(a) for a in [aro_xyz, hph_xyz, hba_xyz, hbd_xyz])
        print(f"plot_3d_residues: {n_total:,} points "
              f"({n_frames} frames × stride {stride}, mask={mask_key})")

        fig.update_layout(
            title=title or f'3D residue map — {mask_key}',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='data',
            ),
            legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01,
                        itemsizing='constant'),
        )
        return fig

    def plot_contact_probability(self, df=None, data: str = 'aromatic_stacking',
                                 title: str = '', add_error_bars: bool = False):
        """Return a Plotly figure of per-residue contact probability.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame to plot. Pass sim.aromatic_contact_probability for aromatic
            contacts or sim.contact_probability for general contacts. When None,
            falls back to sim.contact_probability.
        data : str
            Column in df to plot. Typical values: 'aromatic_stacking',
            'aromatic_pstacking', 'aromatic_tstacking', 'contact_probability',
            or a per-ring column like '0', '1', or 'hydrophobic_contacts'.
        title : str
            Figure title (empty string shows no title).
        add_error_bars : bool
            When True, draws a shaded band using the matching '{data}_error' column.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if go is None:
            raise ImportError("plotly required — install: pip install plotly")
        if df is None:
            df = self.contact_probability
        if df is None:
            raise RuntimeError(
                "Pass a DataFrame or call a compute_*_contact*() method first")
        fig = go.Figure()

        if add_error_bars:
            x_labels = df.index.str.replace('_', ' ')
            fig.add_trace(go.Scatter(
                x=x_labels, y=df[data] + df[f'{data}_error'],
                showlegend=False, mode='lines', line=dict(width=0), name='',
            ))
            fig.add_trace(go.Scatter(
                x=x_labels, y=df[data] - df[f'{data}_error'],
                showlegend=False, mode='lines', line=dict(width=0),
                fillcolor='rgba(226, 226, 226, 0.5)', fill='tonexty', name='',
            ))

        fig.add_trace(go.Scattergl(
            x=df.index.str.replace('_', ' '),
            y=df[data],
            name=data,
            showlegend=True,
        ))
        fig.update_xaxes(
            title='Residues', showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=18), title_font=dict(size=22),
        )
        fig.update_yaxes(
            title='Probability', showgrid=True, gridwidth=1,
            gridcolor='rgb(226, 226, 226)',
            showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=18), title_font=dict(size=22),
        )
        fig.update_layout(
            title=title,
            hovermode='x unified',
            legend=dict(yanchor='bottom', y=0.99, orientation='h'),
        )
        return fig

    def plot_dssp(self, df=None, title: str = '',
                  add_error_bars: bool = True) -> 'go.Figure':
        """Return a Plotly figure of per-residue DSSP helix/sheet probabilities.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame with DSSP columns. Defaults to self.dssp_df.
            Must contain 'DSSP_helix' and/or 'DSSP_sheet' columns.
        title : str
            Figure title.
        add_error_bars : bool
            When True, draw shaded error bands using *_error_up / *_error_low columns.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if go is None:
            raise ImportError("plotly required — install: pip install plotly")
        if df is None:
            df = self.dssp_df
        if df is None:
            raise RuntimeError("Call compute_dssp() first or pass a DataFrame")

        fig = go.Figure()
        x_labels = df.index.str.replace('_', ' ')

        for col, err_up, err_low, color, name in [
            ('DSSP_helix', 'DSSP_helix_error_up', 'DSSP_helix_error_low', '#e377c2', 'Helix'),
            ('DSSP_sheet', 'DSSP_sheet_error_up', 'DSSP_sheet_error_low', '#1f77b4', 'Sheet'),
        ]:
            if col not in df.columns:
                continue
            if add_error_bars and err_up in df.columns and err_low in df.columns:
                fig.add_trace(go.Scatter(
                    x=x_labels, y=df[err_up],
                    mode='lines', line=dict(width=0), showlegend=False, name='',
                ))
                fig.add_trace(go.Scatter(
                    x=x_labels, y=df[err_low],
                    mode='lines', line=dict(width=0), showlegend=False,
                    fill='tonexty', fillcolor='rgba(128,128,128,0.2)', name='',
                ))
            fig.add_trace(go.Scattergl(
                x=x_labels, y=df[col],
                name=name, line=dict(color=color),
            ))

        if 'DSSP_helix_reweighted' in df.columns:
            fig.add_trace(go.Scattergl(
                x=x_labels, y=df['DSSP_helix_reweighted'],
                name='Helix (reweighted)', line=dict(color='#e377c2', dash='dash'),
            ))

        fig.update_xaxes(
            title='Residues', showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=18), title_font=dict(size=22),
        )
        fig.update_yaxes(
            title='Probability', range=[0, 1],
            showgrid=True, gridwidth=1, gridcolor='rgb(226, 226, 226)',
            showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=18), title_font=dict(size=22),
        )
        fig.update_layout(
            title=title,
            hovermode='x unified',
            legend=dict(yanchor='bottom', y=0.99, orientation='h'),
        )
        return fig

    def plot_gyration_salpha_fes(self, df=None, title: str = '',
                                  colorscale: str = 'RdBu_r',
                                  max_dG: float = None,
                                  ncontours: int = 20) -> 'go.Figure':
        """Plot the 2D Rg vs Sα free energy surface as a Plotly filled contour map.

        Parameters
        ----------
        df : pd.DataFrame, optional
            FES DataFrame (rows=Sα, columns=Rg in nm, values in kcal/mol).
            Defaults to self.gyration_salpha_fes_df.
        title : str
            Figure title.
        colorscale : str
            Plotly colorscale. Default 'RdBu_r' (blue=low ΔG).
        max_dG : float, optional
            Clip the color scale at this ΔG value (kcal/mol). Useful for
            focusing contrast on the low-energy basin.
        ncontours : int
            Number of contour levels. Default 20.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if go is None:
            raise ImportError("plotly required — install: pip install plotly")
        if df is None:
            df = self.gyration_salpha_fes_df
        if df is None:
            raise RuntimeError(
                "Call compute_gyration_salpha() first or pass a DataFrame")

        z = df.values.T.copy()   # transpose: rows→Rg, cols→Sα  (x=Sα, y=Rg)
        if max_dG is not None:
            z = np.clip(z, 0.0, float(max_dG))

        fig = go.Figure(go.Contour(
            z=z,
            x=df.index.to_numpy(),     # Sα bin centres (x-axis)
            y=df.columns.to_numpy(),   # Rg bin centres (y-axis, nm)
            colorscale=colorscale,
            ncontours=ncontours,
            contours=dict(showlabels=True, labelfont=dict(size=10)),
            colorbar=dict(title='ΔG (kcal/mol)', titleside='right'),
        ))
        fig.update_xaxes(
            title='S<sub>α</sub>',
            showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=16), title_font=dict(size=20),
        )
        fig.update_yaxes(
            title='R<sub>g</sub> (nm)',
            showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=16), title_font=dict(size=20),
        )
        fig.update_layout(title=title, width=650, height=550)
        return fig

    def plot_contact_map(self, df=None, title: str = '') -> 'go.Figure':
        """Plot the intra-protein residue contact probability map as a heatmap.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Contact probability matrix. Defaults to self.contact_map_df.
        title : str
            Figure title.

        Returns
        -------
        go.Figure
        """
        if go is None:
            raise ImportError("plotly required — install: pip install plotly")
        if df is None:
            if self.contact_map_df is None:
                raise ValueError("Run compute_contact_map() first (distance_raw=False).")
            df = self.contact_map_df

        labels = df.columns.str.replace('_', ' ').tolist()
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=df.values,
            x=labels,
            y=labels,
            colorscale='Jet',
            colorbar=dict(title='Probability'),
        ))
        fig.update_xaxes(
            showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=12), title_font=dict(size=18),
            tickangle=45,
        )
        fig.update_yaxes(
            showline=True, linecolor='black', linewidth=2,
            tickfont=dict(size=12), title_font=dict(size=18),
        )
        fig.update_layout(
            title=title,
            font=dict(size=18),
            plot_bgcolor='rgb(255,255,255)',
            width=700,
            height=650,
        )
        return fig

    def save_plot(self, fig, output_path: str) -> None:
        """Save a Plotly figure as a self-contained HTML file."""
        if go is None:
            raise ImportError("plotly required — install: pip install plotly")
        outstring = '/'.join(output_path.split('/')[:-1])
        print('Saving plot in: ', outstring)
        os.makedirs(outstring, exist_ok=True)
        fig.write_html(output_path)

    # ── Memory management ────────────────────────────────────────────────────

    def clear_results(self, keep_pca_clusters: bool = True) -> None:
        """Free RAM by resetting all computed results to None.

        The trajectory, topology definitions, and ligand chemical features are
        preserved so the object can be reused for further computations (e.g.
        on a new cluster subset) without reloading from disk.

        Parameters
        ----------
        keep_pca_clusters : bool
            When True (default), preserve the PCA cluster frame assignments
            (pca_frames_cl, pca_dtraj, pca_cluster_populations,
            pca_cluster_centers, pca_clusters_fig, pca_silhouette_score)
            so Section-6 cluster analysis can still run after clearing.
            Set to False to wipe PCA results as well.
        """
        # ── Aromatic contacts ─────────────────────────────────────────────
        self._stacked_full         = None
        self._pstacked_full        = None
        self._tstacked_full        = None
        self._stacked_by_ring      = None
        self._aromatic_rings       = None
        self._protein_rings_index  = None
        self.stacking_contacts_dict = None
        self.aromatic_contact_probability = None

        # ── General contact probability ───────────────────────────────────
        self.contact_probability  = None
        self.dual_contact_matrix  = None
        self.kd                   = None
        self.bound_fraction       = None
        self.kd_over_time         = None

        # ── Hydrophobic contacts ──────────────────────────────────────────
        self.hydrophobic_contact_probability      = None
        self.hydrophobic_contact_frames           = None
        self.hydrophobic_atom_contact_probability = None
        self.hydrophobic_ligand_atom_probability  = None
        self.hydrophobic_distances_df             = None

        # ── H-bond contacts ───────────────────────────────────────────────
        self.hbond_contact_probability     = None
        self.hbond_contact_frames_pd       = None
        self.hbond_contact_frames_ld       = None
        self.hbond_pairs_pd                = None
        self.hbond_pairs_ld                = None
        self.hbond_donors_acceptors        = None
        self.hbond_ligand_atom_probability = None
        self.hbond_distances_df            = None
        self.hbond_residue_distances_df    = None

        # ── DSSP ──────────────────────────────────────────────────────────
        self.dssp_df           = None
        self.dssp_over_time_df = None

        # ── Gyration radius & Sα ──────────────────────────────────────────
        self.rg_over_time           = None
        self.salpha_over_time       = None
        self.gyration_salpha_fes_df = None

        # ── Contact map ───────────────────────────────────────────────────
        self.contact_map_df     = None
        self.contact_map_raw_df = None

        # ── SASA ──────────────────────────────────────────────────────────
        self.sasa_atoms_df        = None
        self.sasa_residues_df     = None
        self.sasa_ligand_atoms_df = None

        # ── Residue dot positions ─────────────────────────────────────────
        self.residue_dot_positions = None

        # ── All-atom contacts + PCA feature matrix ────────────────────────
        self.all_atom_contact_probability     = None
        self.all_atom_contact_frames          = None
        self.all_atom_ligand_atom_probability = None
        self.all_atom_distances_df            = None

        # ── Voxel / pharmacophore ─────────────────────────────────────────
        self.negative_space_data          = None
        self.pharmacophore_maps_contested = None
        self.pharmacophore_maps_full      = None
        self.growth_space_features        = None

        # ── Graph clustering ──────────────────────────────────────────────
        self.graph_cluster_distance_matrix = None
        self.graph_cluster_graphs          = None
        self._graph_cluster_traj_attr      = None
        self._graph_cluster_full_n_frames  = None
        self.graph_cluster_labels          = None
        self.graph_cluster_df              = None
        self.graph_cluster_representatives = None
        self.graph_cluster_silhouette      = None
        self.graph_cluster_saved_paths     = None

        # ── PCA results ───────────────────────────────────────────────────
        if not keep_pca_clusters:
            self.pca_result              = None
            self.pca_fes_df              = None
            self.pca_fes_skl_df          = None
            self.pca_dtraj               = None
            self.pca_frames_cl           = None
            self.pca_cluster_centers     = None
            self.pca_clusters_fig        = None
            self.pca_cluster_populations = None
            self.pca_silhouette_score    = None
        else:
            # Only clear the large FES DataFrames and raw projection;
            # cluster assignments and visualisation figures are kept.
            self.pca_result     = None
            self.pca_fes_df     = None
            self.pca_fes_skl_df = None

        import gc
        gc.collect()
        print("Results cleared."
              + (" PCA cluster assignments retained." if keep_pca_clusters else ""))


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute pharmacophores from an MD simulation of an IDP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--topology",   required=True, help="Topology file (.gro or .pdb)")
    p.add_argument("--trajectory", required=True, help="Trajectory file (.xtc or .dcd)")
    p.add_argument("--ligand-resname", default=None,
                   help="Ligand residue name (auto-detected if not provided)")
    p.add_argument("--contact-threshold", type=float, default=DEFAULT_CONTACT_THRESHOLD,
                   help="Minimum contact probability for pharmacophore inclusion")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory for output files")
    p.add_argument("--stride", type=int, default=1,
                   help="Load every N-th trajectory frame")
    p.add_argument("--offset", type=int, default=0,
                   help="Residue number offset for experimental numbering convention")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    os.makedirs(args.output_dir, exist_ok=True)

    sim = PharmacophoreTrajectory(args.topology, args.trajectory)
    sim.load(
        ligand_resname=args.ligand_resname,
        stride=args.stride,
        offset=args.offset,
    )

    # Contact Probabilities
    sim.compute_aromatic_contacts()
    sim.compute_contact_probability()
    sim.compute_hydrophobic_contacts()
    sim.compute_hbond_contacts()
    
    # Pharmacophore definitions
    sim.ligand_align("resname EPI and (name O1 or name C11 or name C18)")
    sim.compute_negative_space()
    sim.define_pharmacophore()
    sim.compute_growth_space_features()


    sim.write_pymol_script(args.output_dir)
    print(f"PyMOL output  → {args.output_dir}")

    # html_path = os.path.join(args.output_dir, "contact_probability.html")
    # fig = sim.plot_contact_probability()
    # sim.save_plot(fig, html_path)
    # print(f"Plotly figure → {html_path}")


if __name__ == "__main__":
    main()
