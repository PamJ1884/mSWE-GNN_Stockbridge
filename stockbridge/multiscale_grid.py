"""Multiscale graph hierarchy for the Stockbridge raster, built by 2x2 cell grouping.

This replaces the MeshKernel-based multiscale meshes of the original mSWE-GNN
(database/graph_creation.py::MultiscaleMesh) for a regular raster domain. The
module only needs numpy, so the hierarchy can be built and checked without
torch; `to_torch` converts the result.

Conventions (same meaning as in the original repository)
---------------------------------------------------------
* Scale 0 is the finest scale. Scales are stacked fine -> coarse in one graph
  with global node indices. `node_ptr[s]:node_ptr[s+1]` are the nodes of scale s.
* `edge_index[:, edge_ptr[s]:edge_ptr[s+1]]` are the intra-scale edges of scale s
  (global indices, both directions). Edge direction is (source, target):
  SWEGNN scatters messages onto `edge_index[1]`.
* `intra_mesh_edge_index[:, intra_edge_ptr[s]:intra_edge_ptr[s+1]]` connects
  scale s (fine) with scale s+1 (coarse). Row 0 is the COARSE node and row 1 the
  FINE node, because MSGNN.forward does
  `col_coarse, row_fine = intra_mesh_edge_index[:, ...]`. There is exactly one
  pair per fine node (its parent), so `len(intra_edge_ptr) == num_scales`.

Hierarchy
---------
* Scale 0: active raster cells (`DEM != DEM.max()`; buildings and cells outside
  the domain carry the artificial maximum and are walls, not nodes), in
  row-major order. `node_ij` keeps the raster (i, j) of every node so that
  predictions can be painted back with `to_raster`.
* Scale s: coarse cell (i // 2**s, j // 2**s). It is a node if at least one of
  its scale-0 descendants is active. Coarse nodes are in row-major order of
  the coarse grid.

Intra-scale edges (design decision)
-----------------------------------
* Scale 0: 4-neighbours between active cells, in both directions.
* Scale s > 0: two coarse cells are connected ONLY IF some edge of scale s-1
  connects one of their children (equivalently, some scale-0 edge connects
  their descendants). Two coarse cells that are geometrically adjacent but
  whose active children are separated by inactive cells (walls) are NOT
  connected, so the coarse scales cannot carry water through buildings.
  As a consequence a coarse scale can have more than one connected component
  even when scale 0 is connected, and vice versa (see the smoke test).

Static attributes (rotation invariant: no slope_x, slope_y, dx, dy)
-------------------------------------------------------------------
* Node area: number of active scale-0 descendants * cell_size**2.
* Node elevation: mean DEM of the active scale-0 descendants. This equals the
  area-weighted mean of the children at scale s-1, so it is consistent with
  how dynamic variables are initialised on coarse nodes (see dataset.py).
  Note: MSGNN pools hidden states with an unweighted mean of the children.
* Node centroid: mean (x, y) of the active scale-0 descendants (cell centres,
  x = (j + 0.5) * cell_size, y = (i + 0.5) * cell_size, i.e. raster row
  direction as +y). Using the active centroid rather than the geometric centre
  of the coarse cell makes edge lengths reflect partially active cells.
* Edge length: distance between the centroids of the two nodes.
* Edge slope: (DEM[source] - DEM[target]) / length, same sign convention as
  `edge_slope` in the original graph_creation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STOCKBRIDGE_CELL_SIZE = 5.0
EXPECTED_NODES_PER_SCALE = (32249, 8162, 2088, 544, 152)


def active_mask_from_dem(dem: np.ndarray) -> np.ndarray:
    """Stockbridge convention: inactive cells carry the artificial maximum DEM.

    Same rule as `active_mask_from_dem` in Stockbridge-SWEGNN
    (experimental/rainfall_aware_swegnn_v2.py), on a 2-D raster.
    """
    dem = np.asarray(dem)
    return dem != dem.max()


@dataclass
class MultiscaleGrid:
    """Stacked multiscale graph. All index arrays use global node indices."""

    num_scales: int
    raster_shape: tuple[int, int]
    cell_size: float
    node_ptr: np.ndarray            # (S+1,)
    edge_ptr: np.ndarray            # (S+1,)
    intra_edge_ptr: np.ndarray      # (S,)
    edge_index: np.ndarray          # (2, E)  int64
    intra_mesh_edge_index: np.ndarray  # (2, E_intra) int64, row 0 coarse, row 1 fine
    node_scale: np.ndarray          # (N,) int64
    node_ij: np.ndarray             # (N, 2) int64, (i, j) in the grid of the node's own scale
    area: np.ndarray                # (N,) float64, m^2
    dem: np.ndarray                 # (N,) float64, m
    centroid: np.ndarray            # (N, 2) float64, m
    edge_length: np.ndarray         # (E,) float64, m
    edge_slope: np.ndarray          # (E,) float64, -
    # ancestors[s][k] = local index at scale s of the ancestor of scale-0 node k
    ancestors: list[np.ndarray] = field(repr=False)
    # flat raster index (i * nx + j) of every scale-0 node
    flat_index0: np.ndarray = field(repr=False)

    @property
    def num_nodes(self) -> int:
        return int(self.node_ptr[-1])

    @property
    def num_edges(self) -> int:
        return int(self.edge_ptr[-1])

    def nodes_per_scale(self) -> list[int]:
        return np.diff(self.node_ptr).tolist()

    def edges_per_scale(self) -> list[int]:
        return np.diff(self.edge_ptr).tolist()

    def scale_slice(self, s: int) -> slice:
        return slice(int(self.node_ptr[s]), int(self.node_ptr[s + 1]))

    def pool_from_scale0(self, values0: np.ndarray) -> np.ndarray:
        """Area-weighted mean of scale-0 values onto every node of the hierarchy.

        values0: (n0,) or (n0, F). Returns (N,) or (N, F); the first n0 rows are
        values0 itself. Every scale-0 cell has the same area, so the
        area-weighted mean of the children equals the plain mean over the
        active scale-0 descendants.
        """
        values0 = np.asarray(values0, dtype=np.float64)
        squeeze = values0.ndim == 1
        if squeeze:
            values0 = values0[:, None]
        n0 = int(self.node_ptr[1])
        if values0.shape[0] != n0:
            raise ValueError(f"Expected {n0} scale-0 rows, got {values0.shape[0]}")
        out = [values0]
        for s in range(1, self.num_scales):
            anc = self.ancestors[s]
            n_s = int(self.node_ptr[s + 1] - self.node_ptr[s])
            count = np.bincount(anc, minlength=n_s).astype(np.float64)
            pooled = np.stack(
                [np.bincount(anc, weights=values0[:, f], minlength=n_s) for f in range(values0.shape[1])],
                axis=1,
            ) / count[:, None]
            out.append(pooled)
        result = np.concatenate(out, axis=0)
        return result[:, 0] if squeeze else result

    def to_raster(self, values: np.ndarray, scale: int = 0, fill: float = np.nan) -> np.ndarray:
        """Paint node values of one scale back on that scale's grid.

        values: (n_s,) values for the nodes of `scale` (local order) or (N,) for
        the whole hierarchy (the scale slice is taken).
        """
        values = np.asarray(values)
        if values.shape[0] == self.num_nodes:
            values = values[self.scale_slice(scale)]
        ny, nx = self.raster_shape
        f = 2 ** scale
        grid = np.full((-(-ny // f), -(-nx // f)), fill, dtype=np.float64)
        ij = self.node_ij[self.scale_slice(scale)]
        grid[ij[:, 0], ij[:, 1]] = values
        return grid


def _four_neighbour_edges(index_grid: np.ndarray) -> np.ndarray:
    """Undirected 4-neighbour pairs between valid entries (>= 0) of an index grid."""
    pairs = []
    for a, b in (
        (index_grid[:, :-1], index_grid[:, 1:]),   # horizontal
        (index_grid[:-1, :], index_grid[1:, :]),   # vertical
    ):
        keep = (a >= 0) & (b >= 0)
        pairs.append(np.stack([a[keep], b[keep]]))
    return np.concatenate(pairs, axis=1)


def _both_directions_sorted(pairs: np.ndarray) -> np.ndarray:
    """Add reverse direction, drop duplicates and sort by (source, target)."""
    directed = np.concatenate([pairs, pairs[::-1]], axis=1)
    directed = np.unique(directed, axis=1)  # lexicographic sort on (row 0, row 1)
    return directed


def build_multiscale_grid(
    dem: np.ndarray,
    num_scales: int = 4,
    cell_size: float = STOCKBRIDGE_CELL_SIZE,
    active: np.ndarray | None = None,
) -> MultiscaleGrid:
    """Build the stacked multiscale graph from a 2-D DEM raster.

    dem: (ny, nx) elevation raster.
    num_scales: number of scales including scale 0 (>= 2).
    cell_size: raster cell size in m (Stockbridge: 5 m).
    active: optional boolean mask; defaults to `active_mask_from_dem(dem)`.
    """
    dem = np.asarray(dem, dtype=np.float64)
    if dem.ndim != 2:
        raise ValueError(f"DEM must be 2-D, got shape {dem.shape}")
    if num_scales < 2:
        raise ValueError("num_scales must be >= 2")
    if active is None:
        active = active_mask_from_dem(dem)
    active = np.asarray(active, dtype=bool)
    ny, nx = dem.shape

    # ---- scale 0 -------------------------------------------------------------
    ii0, jj0 = np.nonzero(active)          # row-major order
    n0 = ii0.size
    flat_index0 = ii0 * nx + jj0
    dem0 = dem[ii0, jj0]
    xy0 = np.stack([(jj0 + 0.5) * cell_size, (ii0 + 0.5) * cell_size], axis=1)

    index_grid0 = np.full((ny, nx), -1, dtype=np.int64)
    index_grid0[ii0, jj0] = np.arange(n0)
    local_edges = [_both_directions_sorted(_four_neighbour_edges(index_grid0))]

    node_ij = [np.stack([ii0, jj0], axis=1)]
    count0 = [np.ones(n0)]
    dem_s = [dem0]
    xy_s = [xy0]
    ancestors = [np.arange(n0)]
    parents = []  # parents[s][k] = local index at scale s+1 of node k of scale s

    # ---- coarse scales ------------------------------------------------------
    for s in range(1, num_scales):
        f = 2 ** s
        cny, cnx = -(-ny // f), -(-nx // f)
        ci, cj = ii0 // f, jj0 // f
        coarse_flat = ci * cnx + cj
        # unique coarse cells in row-major order; inverse = ancestor of each scale-0 node
        uniq, anc = np.unique(coarse_flat, return_inverse=True)
        n_s = uniq.size
        node_ij.append(np.stack([uniq // cnx, uniq % cnx], axis=1))
        cnt = np.bincount(anc, minlength=n_s).astype(np.float64)
        count0.append(cnt)
        dem_s.append(np.bincount(anc, weights=dem0, minlength=n_s) / cnt)
        xy_s.append(np.stack([np.bincount(anc, weights=xy0[:, d], minlength=n_s) / cnt for d in range(2)], axis=1))
        ancestors.append(anc)

        # parent of each node of scale s-1: any scale-0 descendant gives the same answer
        prev_anc = ancestors[s - 1]
        n_prev = node_ij[s - 1].shape[0]
        parent = np.full(n_prev, -1, dtype=np.int64)
        parent[prev_anc] = anc
        if (parent < 0).any():
            raise RuntimeError(f"Scale {s-1} has nodes without a parent")
        # consistency: all descendants of one fine node share the coarse ancestor
        if not np.array_equal(parent[prev_anc], anc):
            raise RuntimeError(f"Inconsistent parent mapping between scales {s-1} and {s}")
        parents.append(parent)

        # coarse edges: images of the fine edges whose endpoints have different parents
        fine_edges = local_edges[s - 1]
        coarse_pairs = parent[fine_edges]
        coarse_pairs = coarse_pairs[:, coarse_pairs[0] != coarse_pairs[1]]
        local_edges.append(_both_directions_sorted(coarse_pairs))

    # ---- stack ---------------------------------------------------------------
    sizes = [ij.shape[0] for ij in node_ij]
    node_ptr = np.cumsum([0] + sizes).astype(np.int64)
    edge_ptr = np.cumsum([0] + [e.shape[1] for e in local_edges]).astype(np.int64)
    edge_index = np.concatenate([e + node_ptr[s] for s, e in enumerate(local_edges)], axis=1).astype(np.int64)

    intra_blocks = []
    for s, parent in enumerate(parents):
        fine = np.arange(parent.size) + node_ptr[s]
        coarse = parent + node_ptr[s + 1]
        intra_blocks.append(np.stack([coarse, fine]))  # row 0 coarse, row 1 fine
    intra_edge_ptr = np.cumsum([0] + [b.shape[1] for b in intra_blocks]).astype(np.int64)
    intra_mesh_edge_index = np.concatenate(intra_blocks, axis=1).astype(np.int64)

    area = np.concatenate(count0) * cell_size ** 2
    dem_all = np.concatenate(dem_s)
    centroid = np.concatenate(xy_s, axis=0)
    node_scale = np.concatenate([np.full(n, s, dtype=np.int64) for s, n in enumerate(sizes)])

    src, dst = edge_index
    edge_length = np.linalg.norm(centroid[dst] - centroid[src], axis=1)
    if (edge_length <= 0).any():
        raise RuntimeError("Found an edge of zero length")
    edge_slope = (dem_all[src] - dem_all[dst]) / edge_length

    return MultiscaleGrid(
        num_scales=num_scales,
        raster_shape=(ny, nx),
        cell_size=float(cell_size),
        node_ptr=node_ptr,
        edge_ptr=edge_ptr,
        intra_edge_ptr=intra_edge_ptr,
        edge_index=edge_index,
        intra_mesh_edge_index=intra_mesh_edge_index,
        node_scale=node_scale,
        node_ij=np.concatenate(node_ij, axis=0).astype(np.int64),
        area=area,
        dem=dem_all,
        centroid=centroid,
        edge_length=edge_length,
        edge_slope=edge_slope,
        ancestors=ancestors,
        flat_index0=flat_index0.astype(np.int64),
    )


# ---- checks used by the smoke test ------------------------------------------

def connected_components_per_scale(grid: MultiscaleGrid) -> list[int]:
    """Number of connected components of the intra-scale graph at every scale."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    counts = []
    for s in range(grid.num_scales):
        lo = int(grid.node_ptr[s])
        n = int(grid.node_ptr[s + 1]) - lo
        e = grid.edge_index[:, grid.edge_ptr[s]:grid.edge_ptr[s + 1]] - lo
        adj = coo_matrix((np.ones(e.shape[1]), (e[0], e[1])), shape=(n, n))
        counts.append(int(connected_components(adj, directed=False)[0]))
    return counts


def check_hierarchy(grid: MultiscaleGrid) -> dict:
    """Structural checks. Returns a dict of booleans/values; raises nothing."""
    out = {}
    S = grid.num_scales
    # every fine node has exactly one parent, and parents are in the next scale
    one_parent = []
    for s in range(S - 1):
        block = grid.intra_mesh_edge_index[:, grid.intra_edge_ptr[s]:grid.intra_edge_ptr[s + 1]]
        coarse, fine = block
        n_fine = int(grid.node_ptr[s + 1] - grid.node_ptr[s])
        fine_counts = np.bincount(fine - grid.node_ptr[s], minlength=n_fine)
        in_range = (
            (fine >= grid.node_ptr[s]).all() and (fine < grid.node_ptr[s + 1]).all()
            and (coarse >= grid.node_ptr[s + 1]).all() and (coarse < grid.node_ptr[s + 2]).all()
        )
        every_coarse_has_child = np.unique(coarse).size == int(grid.node_ptr[s + 2] - grid.node_ptr[s + 1])
        one_parent.append(bool(in_range and (fine_counts == 1).all() and every_coarse_has_child))
    out["exactly_one_parent_per_fine_node"] = one_parent
    # intra-scale edges stay inside their scale, both directions present, no self loops
    within, symmetric = [], []
    for s in range(S):
        e = grid.edge_index[:, grid.edge_ptr[s]:grid.edge_ptr[s + 1]]
        within.append(bool(((e >= grid.node_ptr[s]) & (e < grid.node_ptr[s + 1])).all()))
        fwd = set(map(tuple, e.T.tolist()))
        symmetric.append(all((b, a) in fwd for a, b in fwd) and all(a != b for a, b in fwd))
    out["edges_within_scale"] = within
    out["edges_symmetric_no_self_loops"] = symmetric
    # area conservation: every scale covers the same active area
    out["area_per_scale_m2"] = [float(grid.area[grid.scale_slice(s)].sum()) for s in range(S)]
    return out
