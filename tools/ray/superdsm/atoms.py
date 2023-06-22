from .output import get_output

import numpy as np
import skimage.morphology as morph
import skimage.segmentation


def _find_seed_of_region(region, seeds):
    assert isinstance(region, np.ndarray)
    assert str(region.dtype) == 'bool'
    candidates = list()
    for seed in seeds:
        seed = tuple(seed)
        if region[seed]: candidates.append(seed)
    assert len(candidates) == 1, f'There is no (unique) seed. Number of possible seeds: {len(candidates)}'
    return candidates[0]


class AtomAdjacencyGraph:
    """Graph representation of the adjacencies of atomic image regions.

    This corresponds to the adjacency graph :math:`\\mathcal G` as defined in :ref:`pipeline_theory_c2freganal`.

    :param atoms: Integer-valued image representing the universe of atomic image regions. Each atomic image region has a unique label, which is the integer value.
    :param clusters: Integer-valued image representing the regions of possibly clustered obejcts. Each region has a unique label, which is the integer value.
    :param fg_mask: Binary image corresponding to a rough representation of the image foreground. This means that an image point :math:`x \\in \\Omega` is ``True`` if :math:`Y_\\omega|_{\\omega=\\{x\\}} > 0` and ``False`` otherwise.
    :param seeds: The seed points which were used to determine the atomic image regions, represented by a list of tuples of coordinates. The :ref:`pipeline` only uses these for rendering the adjacency graph (see the :py:meth:`~.get_edge_lines` method).
    :param out: An instance of an :py:class:`~superdsm.output.Output` sub-class, ``'muted'`` if no output should be produced, or ``None`` if the default output should be used.

    .. runblock:: pycon

       >>> import superdsm.atoms
       >>> import numpy as np
       >>> atoms = np.array([[1, 1, 2, 4],
       ...                   [1, 3, 2, 4],
       ...                   [3, 3, 3, 4]])
       >>> clusters = np.array([[1, 1, 2, 2],
       ...                      [1, 2, 2, 2],
       ...                      [2, 2, 2, 2]])
       >>> fg_mask = np.array([[True, False, True, False],
       ...                     [True, False, True,  True],
       ...                     [True,  True, True,  True]])
       >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
       >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
       >>> adj[1]
       >>> adj[2]
       >>> adj[3]
       >>> adj[4]
    """

    def __init__(self, atoms, clusters, fg_mask, seeds, out=None):
        out = get_output(out)
        self._adjacencies, se = {atom_label: set() for atom_label in range(1, atoms.max() + 1)}, morph.disk(1)
        self._atoms_by_cluster = dict()
        self._cluster_by_atom  = dict()
        self._seeds            = dict()
        for l0 in range(1, atoms.max() + 1):
            cc = (atoms == l0)
            if not cc.any(): continue
            cluster_label = clusters[cc][0]
            cluster_mask  = np.logical_and(fg_mask, clusters == cluster_label)
            cc_dilated = np.logical_and(morph.binary_dilation(cc, se), np.logical_not(cc))
            cc_dilated = np.logical_and(cc_dilated, cluster_mask)
            if cluster_label not in self._atoms_by_cluster:
                self._atoms_by_cluster[cluster_label] = set()
            adjacencies = set(atoms[cc_dilated].flatten()) - {0, l0}
            self._adjacencies[l0] |= adjacencies
            for l1 in adjacencies:
                self._adjacencies[l1] |= {l0}
            self._cluster_by_atom[l0] = cluster_label
            self._atoms_by_cluster[cluster_label] |= {l0}
            self._seeds[l0] = _find_seed_of_region(atoms == l0, seeds)
            out.intermediate('Processed atom %d / %d' % (l0, atoms.max()))
        out.write('Computed adjacency graph')
        assert self._is_symmetric()
    
    def __getitem__(self, atom_label):
        return self._adjacencies[atom_label]

    def _update_clusters(self, atom_label):
        old_cluster_label = self._cluster_by_atom[atom_label]
        if len(self[atom_label]) == 0 and len(self._atoms_by_cluster[old_cluster_label]) > 1:
            new_cluster_label = max(self.cluster_labels) + 1
            self._cluster_by_atom[atom_label] = new_cluster_label
            self._atoms_by_cluster[new_cluster_label]  = {atom_label}
            self._atoms_by_cluster[old_cluster_label] -= {atom_label}
    
    def get_cluster_label(self, atom_label):
        """Returns the label of the region of possibly clustered objects, which an atomic image region is a part of.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.ones(atoms.shape, bool)
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.get_cluster_label(1)
           >>> adj.get_cluster_label(2)
           >>> adj.get_cluster_label(3)
           >>> adj.get_cluster_label(4)
        """
        return self._cluster_by_atom[atom_label]
    
    def get_atoms_in_cluster(self, cluster_label):
        """Returns the set of labels of all atomic image regions, which are part of a region of possibly clustered objects.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.ones(atoms.shape, bool)
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.get_atoms_in_cluster(1)
           >>> adj.get_atoms_in_cluster(2)
        """
        return self._atoms_by_cluster[cluster_label]
    
    @property
    def cluster_labels(self):
        """The set of labels of all regions of possibly clustered objects.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.ones(atoms.shape, bool)
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.cluster_labels
        """
        return frozenset(self._atoms_by_cluster.keys())
    
    @property
    def atom_labels(self):
        """The set of labels of all atomic image regions.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.ones(atoms.shape, bool)
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.atom_labels
        """
        return frozenset(self._cluster_by_atom.keys())

    def get_seed(self, atom_label):
        """Returns the seed point which was used to determine an atomic image region.
        
        :return: Tuple of the coordinates of the seed point.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.ones(atoms.shape, bool)
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.get_seed(1)
           >>> adj.get_seed(2)
           >>> adj.get_seed(3)
           >>> adj.get_seed(4)
        """
        return self._seeds[atom_label]
    
    def get_edge_lines(self, accept='all', reduce=True):
        """Returns a list of lines corresponding to the edges of the graph.

        :param accept: Must be either ``all`` or a callable. If ``all`` is used, all edges of the graph are included. Otherwise, an edge ``(i,j)`` is included only if ``accept(i)`` and ``accept(j)`` evaluate to ``True``, where ``i`` and ``j`` are the labels of two adjacent atomic image regions.
        :param reduce: If ``True``, then an edge ``(i,j)`` is included only if ``i > j``. Otherwise, both edges ``(i,j)`` and ``(j,i)`` are included.

        Each line is a tuple of two seed points, and each seed point is a tuple of coordinates.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.array([[True, False, True, False],
           ...                     [True, False, True,  True],
           ...                     [True,  True, True,  True]])
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.get_edge_lines()
           >>> adj.get_edge_lines(lambda i: i != 4)
           >>> adj.get_edge_lines(lambda i: i != 4, reduce=False)
        """
        if isinstance(accept, str) and accept == 'all': accept = lambda atom_label: True
        assert callable(accept), f'Not a callable: {str(accept)}'
        lines = []
        for l in self.atom_labels:
            seed_l = self.get_seed(l)
            if not accept(l): continue
            for k in self[l]:
                seed_k = self.get_seed(k)
                if not accept(k): continue
                if reduce and l > k: continue
                lines.append((seed_l, seed_k))
        return lines

    @property
    def max_degree(self):
        """The maximum degree of the graph.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.array([[True, False, True, False],
           ...                     [True, False, True,  True],
           ...                     [True,  True, True,  True]])
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.max_degree
        """
        return max(self.get_atom_degree(atom_label) for atom_label in self.atom_labels)

    def get_atom_degree(self, atom_label):
        """Returns the number of adjacent atomic image regions.

        .. runblock:: pycon

           >>> import superdsm.atoms
           >>> import numpy as np
           >>> atoms = np.array([[1, 1, 2, 4],
           ...                   [1, 3, 2, 4],
           ...                   [3, 3, 3, 4]])
           >>> clusters = np.array([[1, 1, 2, 2],
           ...                      [1, 2, 2, 2],
           ...                      [2, 2, 2, 2]])
           >>> fg_mask = np.array([[True, False, True, False],
           ...                     [True, False, True,  True],
           ...                     [True,  True, True,  True]])
           >>> seeds = [(0, 0), (0, 2), (2, 1), (1, 3)]
           >>> adj = superdsm.atoms.AtomAdjacencyGraph(atoms, clusters, fg_mask, seeds, 'muted')
           >>> adj.get_atom_degree(1)
           >>> adj.get_atom_degree(2)
           >>> adj.get_atom_degree(3)
           >>> adj.get_atom_degree(4)
        """
        return len(self[atom_label])

    def _is_symmetric(self):
        for atom1 in self.atom_labels:
            if not all(atom1 in self[atom2] for atom2 in self[atom1]):
                return False
        return True

