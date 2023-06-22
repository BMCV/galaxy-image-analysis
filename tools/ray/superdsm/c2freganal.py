from .pipeline import Stage
from ._aux import get_ray_1by1, copy_dict
from .objects import cvxprog, Object
from .atoms import AtomAdjacencyGraph
from .image import Image

import ray
import scipy.ndimage as ndi
import numpy as np
import skimage.segmentation as segm
import skimage.morphology as morph
import queue, contextlib, math, hashlib


def _get_next_seed(region, where, score_func, connectivity=4):
    if   connectivity == 4: footprint = morph.disk(1)
    elif connectivity == 8: footprint = np.ones((3,3))
    else: raise ValeError(f'unknown connectivity: {connectivity}')
    mask  = np.logical_and(region.mask, where)
    image = region.model
    image_max = ndi.maximum_filter(image, footprint=footprint)
    max_mask  = np.logical_and(image_max == image, mask)
    if max_mask.any():
        maxima = ndi.label(max_mask)[0]
        maxima_labels = frozenset(maxima.reshape(-1)) - {0}
        scores = {max_label: score_func(maxima == max_label) for max_label in maxima_labels}
        label  = max(maxima_labels, key=scores.get)
        if scores[label] > -np.inf: return (maxima == label)
    return None


def _watershed_split(region, *markers):
    markers_map = np.zeros(region.model.shape, int)
    for marker_label, marker in enumerate(markers, start=1):
        assert markers_map[marker] == 0
        markers_map[marker] = marker_label
    watershed = segm.watershed(region.model.max() - region.model.clip(0, np.inf), markers=markers_map, mask=region.mask)
    return [watershed == marker_label for marker_label in range(1, len(markers) + 1)]


def _normalize_labels_map(labels, first_label=0, skip_labels=[]):
    result = np.zeros_like(labels)
    label_translation = {}
    next_label = first_label
    for old_label in sorted(np.unique(labels.reshape(-1))):
        if old_label in skip_labels: continue
        result[ labels == old_label] = next_label
        label_translation[old_label] = next_label
        next_label += 1
    return result, label_translation


def _hash_mask(mask):
    mask = mask.astype(np.uint8)
    return hashlib.sha1(mask).digest()


def _get_cached_normalized_energy_computer(y, cluster):
    cache = dict()
    cp_buffer = Image(model=y.model, mask=np.zeros(cluster.full_mask.shape, bool))
    def compute_normalized_energy(obj, region, atoms_map, dsm_cfg):
        cp_kwargs = copy_dict(dsm_cfg)
        cp_kwargs.pop('smooth_mat_max_allocations', None)
        cp_region = obj.get_cvxprog_region(region, atoms_map, cp_kwargs.pop('background_margin'))
        cp_region_hash = _hash_mask(cp_region.mask)
        cache_hit = cache.get(cp_region_hash, None)
        if cache_hit is None:
            if (cp_region.model[cp_region.mask] > 0).all() or (cp_region.model[cp_region.mask] < 0).all():
                energy = None
            else:
                cp_buffer.mask[cluster.full_mask] = cp_region.mask[cluster.mask]
                with contextlib.redirect_stdout(None):
                    J, model, status = cvxprog(cp_buffer, smooth_mat_allocation_lock=None, **cp_kwargs)
                cp_buffer.mask[cluster.full_mask].fill(False)
                energy = J(model)
            cache_hit = energy / cp_region.mask.sum()
            cache[cp_region_hash] = cache_hit
        return cache_hit
    return compute_normalized_energy


class C2F_RegionAnalysis(Stage):
    """Implements the :ref:`pipeline_theory_c2freganal` scheme.

    This stage requires ``y`` and ``dsm_cfg`` for input and produces ``y_mask``, ``atoms``, ``adjacencies``, ``seeds``, ``clusters`` for output. Refer to :ref:`pipeline_inputs_and_outputs` for more information on the available inputs and outputs.

    Hyperparameters
    ---------------

    The following hyperparameters can be used to control this pipeline stage:

    ``c2f-region-analysis/seed_connectivity``
        Image points which are adjacent to each other are *not* to determine the atomic image regions. Adjacency of such image points is determined using either 4-connectivity or 8-connectivity. Must be either 4 or 8. Defaults to 8.

    ``c2f-region-analysis/min_atom_radius``
        No region determined by the :ref:`pipeline_theory_c2freganal` scheme is smaller than a circle of this radius (in terms of the surface area). Defaults to 15, or to ``AF_min_atom_radius × radius`` if configured automatically (and ``AF_min_atom_radius`` defaults to 0.33).

    ``c2f-region-analysis/max_atom_norm_energy``
        No atomic image region :math:`\\omega` determined by the :ref:`pipeline_theory_c2freganal` has a normalized energy :math:`r(\\omega)` smaller than this value. Corresponds to *max_norm_energy1* in the :ref:`paper <references>` (Supplemental Materials 5 and 8). Defaults to 0.05.

    ``c2f-region-analysis/min_norm_energy_improvement``
        Each split performed during the computation of the atomic image regions must improve the normalized energy :math:`r(\\omega)` of an image region :math:`\\omega` by at least this factor (see :ref:`pipeline_theory_c2freganal`). Given that an image region is split into the sub-regions :math:`\\omega_1, \\omega_2`, the improvement of the split is defined by the fraction :math:`\\max\\{ r(\\omega_1), r(\\omega_1) \\} / r(\\omega_1 \\cup \\omega_2)`. Lower values of the fraction correspond to better improvements. Defaults to 0.1.

    ``c2f-region-analysis/max_cluster_marker_irregularity``
        Threshold for the "irregularity" of image regions, which is used to determine the output ``y_mask``. Image regions with an "irregularity" higher than this value are masked as "empty" image regions and discarded from further considerations. This is the threshold for the P/A ratio described in the :ref:`paper <references>` (see Section 3.1 and *max_pa_ratio* in Supplemental Material 8). Defaults to 0.2.

    .. note::
       This stage takes the DSM-related hyperparameters as an input. Due to a bug in the original implementation, the value set for the hyperparameter ``dsm/background_margin`` was disrespected and a value of 20 was always used instead. However, the impact on the results is only subtle. Having this issue fixed, the results are mostly consistent with those originally reported in the :ref:`paper <references>` (the hyperparameter :math:`\\alpha` is changed from 0.1 to 0.2 for the GOWT1-2 dataset when using SuperDSM*, see the ``examples/GOWT1-2/default/adapted/task.json`` file).
    """

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(C2F_RegionAnalysis, self).__init__('c2f-region-analysis',
                                                 inputs  = ['y', 'dsm_cfg'],
                                                 outputs = ['y_mask', 'atoms', 'adjacencies', 'seeds', 'clusters'])

    def process(self, input_data, cfg, out, log_root_dir):
        seed_connectivity = cfg.get('seed_connectivity', 8)
        min_atom_radius = cfg.get('min_atom_radius', 15)
        max_atom_norm_energy = cfg.get('max_atom_norm_energy', 0.05)
        min_norm_energy_improvement = cfg.get('min_norm_energy_improvement', 0.1)
        max_cluster_marker_irregularity = cfg.get('max_cluster_marker_irregularity', 0.2)

        dsm_cfg = copy_dict(input_data['dsm_cfg'])
        dsm_cfg['smooth_amount'] = np.inf
        
        out.intermediate(f'Analyzing cluster markers...')
        y = Image.create_from_array(input_data['y'], normalize=False)
        fg_mask = (y.model > 0)
        fg_bd   = np.logical_xor(fg_mask, morph.binary_erosion(fg_mask, morph.disk(1)))
        y_mask  = np.ones(y.model.shape, bool)
        cluster_markers = ndi.label(fg_mask)[0]
        for cluster_marker_label in np.unique(cluster_markers):
            cluster_marker = (cluster_markers == cluster_marker_label)
            cluster_marker_irregularity = fg_bd[cluster_marker].sum() / cluster_marker.sum()
            if cluster_marker_irregularity > max_cluster_marker_irregularity:
                y_mask[cluster_marker] = False
                
        cluster_markers[~y_mask] = cluster_markers.min()
        cluster_markers = _normalize_labels_map(cluster_markers, first_label=0)[0]
        out.write(f'Extracted {cluster_markers.max()} cluster markers')
        
        clusters  = segm.watershed(ndi.distance_transform_edt(cluster_markers == 0), markers=cluster_markers)
        atoms_map = np.full(y.model.shape, 0)
        atom_candidate_by_label = {}
        
        y_id = ray.put(y)
        dsm_cfg_id = ray.put(dsm_cfg)
        y_mask_id = ray.put(y_mask)
        clusters_id = ray.put(clusters)
        futures = [process_cluster.remote(clusters_id, cluster_label, y_id, y_mask_id, max_atom_norm_energy, min_atom_radius, min_norm_energy_improvement, dsm_cfg_id, seed_connectivity) for cluster_label in frozenset(clusters.reshape(-1)) - {0}]
        max_normalized_energy = -np.inf
        for ret_idx, ret in enumerate(get_ray_1by1(futures)):
            cluster_label, cluster_universe, cluster_atoms, cluster_atoms_map, cluster_max_normalized_energy = ret
            cluster_label_offset = atoms_map.max()
            max_normalized_energy = max((cluster_max_normalized_energy, max_normalized_energy))
            cluster = y.get_region(clusters == cluster_label, shrink=True)
            atoms_map[cluster.full_mask] = cluster_label_offset + cluster_atoms_map[cluster.mask]
            for atom_candidate in cluster_atoms:
                atom_candidate_by_label[cluster_label_offset + list(atom_candidate.footprint)[0]] = atom_candidate
                atom_candidate.seed = np.round(ndi.center_of_mass(atom_candidate.seed)).astype(int) + cluster.offset
            out.intermediate(f'Analyzing clusters... {ret_idx + 1} / {len(futures)}')
            
        atoms_map, label_translation = _normalize_labels_map(atoms_map, first_label=1, skip_labels=[0])
        for old_label, atom_candidate in dict(atom_candidate_by_label).items():
            atom_candidate_by_label[label_translation[old_label]] = atom_candidate
        out.write(f'Extracted {atoms_map.max()} atoms (max energy rate: {max_normalized_energy:g})')
        
        # Compute adjacencies graph
        atom_nodes  = [atom_candidate_by_label[atom_label].seed for atom_label in sorted(label_translation.values())]
        adjacencies = AtomAdjacencyGraph(atoms_map, clusters, fg_mask, atom_nodes, out)
        
        return {
            'y_mask': y_mask,
            'atoms': atoms_map,
            'adjacencies': adjacencies,
            'seeds': atom_nodes,
            'clusters': clusters
        }

    def configure_ex(self, scale, radius, diameter):
        return {
            'min_atom_radius': (radius, 0.33, dict(type=int)),
        }


@ray.remote
def process_cluster(*args, **kwargs):
    return _process_cluster_impl(*args, **kwargs)


def _process_cluster_impl(clusters, cluster_label, y, y_mask, max_atom_norm_energy, min_atom_radius, min_norm_energy_improvement, dsm_cfg, seed_connectivity):
    min_atom_size = math.pi * (min_atom_radius ** 2)
    cluster = y.get_region(clusters == cluster_label, shrink=True)
    masked_cluster = cluster.get_region(cluster.shrink_mask(y_mask))
    root_candidate = Object()
    root_candidate.footprint = frozenset([1])
    root_candidate.seed = _get_next_seed(masked_cluster, cluster.model > 0, lambda loc: cluster.model[loc].max(), seed_connectivity)
    atoms_map = cluster.mask.astype(int) * list(root_candidate.footprint)[0]
    compute_normalized_energy = _get_cached_normalized_energy_computer(y, cluster)

    leaf_candidates = []
    split_queue = queue.Queue()
    root_candidate.normalized_energy = compute_normalized_energy(root_candidate, masked_cluster, atoms_map, dsm_cfg)
    if root_candidate.normalized_energy > max_atom_norm_energy:
        split_queue.put(root_candidate)
    else:
        leaf_candidates.append(root_candidate)

    seed_distances = ndi.distance_transform_edt(~root_candidate.seed)
    while not split_queue.empty():
        c0 = split_queue.get()
        c0_mask = c0.get_mask(atoms_map)
        
        if c0_mask.sum() < 2 * min_atom_size:
            leaf_candidates.append(c0) ## the region is too small to be split
            continue

        c1 = Object()
        c2 = Object()
        c1.seed = c0.seed
        c2.seed = _get_next_seed(masked_cluster, np.all((cluster.model > 0, c0_mask, seed_distances >= 1), axis=0), lambda loc: seed_distances[loc].max(), seed_connectivity)
        assert not np.logical_and(c1.seed, c2.seed).any()
        if c2.seed is None:
            leaf_candidates.append(c0)
            continue
        else:
            seed_distances = np.min([seed_distances, ndi.distance_transform_edt(~c2.seed)], axis=0)

        new_atom_label   = atoms_map.max() + 1
        c1_mask, c2_mask = _watershed_split(cluster.get_region(c0_mask), c1.seed, c2.seed)
            
        if c1_mask.sum() < min_atom_size:
            c0.seed = c2.seed   ## change the seed for current region…
            split_queue.put(c0) ## …and try again with different seed
            continue
            
        if c2_mask.sum() < min_atom_size:
            split_queue.put(c0) ## try again with different seed
            continue
            
        atoms_map_previous = atoms_map.copy()
        atoms_map[c2_mask] = new_atom_label
        c1.footprint = frozenset(c0.footprint)
        c2.footprint = frozenset([new_atom_label])
        assert c1_mask[cluster.mask].any() and not np.logical_and(~cluster.mask, c1_mask).any()
        assert c2_mask[cluster.mask].any() and not np.logical_and(~cluster.mask, c2_mask).any()

        for c in (c1, c2):
            try:
                c.normalized_energy = compute_normalized_energy(c, masked_cluster, atoms_map, dsm_cfg)
            except:
                c.normalized_energy = None
                
        if c1.normalized_energy is None and c2.normalized_energy is None:
            split_queue.put(c0) ## try again with different seed
            atoms_map = atoms_map_previous
            continue
            
        if c1.normalized_energy is None and c2.normalized_energy is not None:
            c0.seed = c2.seed   ## change the seed for current region…
            split_queue.put(c0) ## …and try again with different seed
            atoms_map = atoms_map_previous
            continue
            
        if c1.normalized_energy is not None and c2.normalized_energy is None:
            split_queue.put(c0) ## try again with different seed
            atoms_map = atoms_map_previous
            continue
            
        assert c1.normalized_energy is not None and c2.normalized_energy is not None, str((c1.normalized_energy, c2.normalized_energy))
        norm_energy_improvement = 1 - max((c1.normalized_energy, c2.normalized_energy)) / c0.normalized_energy
        if norm_energy_improvement < min_norm_energy_improvement:
            split_queue.put(c0) ## try again with different seed
            atoms_map = atoms_map_previous
        else:
            for c in (c1, c2):
                if c.normalized_energy > max_atom_norm_energy:
                    split_queue.put(c)
                else:
                    leaf_candidates.append(c)

    root_candidate.footprint = frozenset(atoms_map.reshape(-1)) - {0}
    assert frozenset([list(c.footprint)[0] for c in leaf_candidates]) == root_candidate.footprint
    max_normalized_energy = max((c.normalized_energy for c in leaf_candidates))
    return cluster_label, root_candidate, leaf_candidates, atoms_map, max_normalized_energy
    
