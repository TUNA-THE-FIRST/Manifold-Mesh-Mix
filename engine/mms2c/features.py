"""
Find curved surfaces in a triangle mesh so they can be rebuilt as true curves.

The mesh is split into patches, each lying on one simple surface:

  cylinder  round holes, pins, straight rounded edges (fillets)
  cone      countersinks, chamfers around round edges, cone tips
  sphere    domes, dimples, balls, rounded corners where three edges meet
  torus     rounded edges that follow a curve (the corners of a rounded box lid,
            a fillet around the base of a pin, a rounded rim on a hole)

How a patch is found: take two or three neighbouring facets, work out the one
surface they could lie on, then grow outwards while the facets' corners stay on
that surface. Straight chamfers and flat faces need nothing: they are already
exact planes.

A patch is only accepted if its outline follows the surface's natural boundary
lines (for a cylinder: the two end circles and the two straight sides). This
rejects partial matches, holes crossed by other features, oblique cuts and so on;
those areas simply stay faceted.

Each accepted patch records its exact surface and its boundary lines; build.py
turns it into one true curved face and stitches it to the flat faces around it.
"""
import copy
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from . import progress
from .accuracy import scale
ACCURACY = scale("features")

import numpy as np

TOL = 0.001          # mm: CAD exports put mesh corners on the true surface to within this
FLOAT_TOL = 2e-7     # plus this fraction of the largest coordinate (STL stores 32-bit floats)
WELD_GRID = 1e-4     # mm: corners on one cell of this grid are joined when the file is read (see load_stl)
MIRROR = False          # search one side of a mirror-symmetric part, the other by mirroring (off: see CLAUDE.md)
MIRROR_SHARE = 0.5      # a mirrored surface takes the facets on it if they cover this share of its own area
NORMAL_DEG = 30      # a facet may tilt this far from the true surface normal (coarse meshes)
SMOOTH_DEG = 40      # facets meeting at a gentler bend than this can be one curved surface
COPLANAR_DEG = 0.05  # triangles closer to flat than this are merged into one facet
MAX_SAG = 0.25       # mm: the middle of a facet may sit this far inside the curve (coarse meshes)
BEND_SLACK_DEG = 8   # neighbouring facets must bend by what the surface predicts, within this
MIN_SPAN_DEG = 15    # a partial patch must curve through at least this much
EXACT_CARVE_MIN = 100       # facets: a cylinder, cone, torus or sphere patch this big blamed for trouble gives up only its facets near it
PINHOLE_FACETS = 4          # facets: a hole this small closed in by one patch, on its corners, is the patch's
SAME_FACETS = 30            # a cylinder, torus or sphere found over this many facets ...
SAME_TOL = 0.003 * ACCURACY   # mm: ... takes in a strip beside it whose corners lie this near it,
SAME_TILT_DEG = 10          # ... fitted as nearly the same surface: axes within this,
SAME_SHARE = 0.15           # ... radii (and sphere centres) within this share
JOINT_STRAY = 0.25          # two such pieces become one if a surface fitted to both holds all
                            # their facets to the mesh tolerance but this share of the smaller's
CORNER_FIT_FACETS = 8       # a smooth area of at least this many facets ...
CORNER_FIT_DEV = 0.005 * ACCURACY  # mm: ... every corner this near a cylinder fitted to them is
                            # one (at 0.02 the GPS case back's tapped thread failed: 1,502 -> 3,032)
CORNER_FIT_MAX_R = 1000.0   # mm: ... of at most this radius (more is a plane)
LAND_TOL = 0.002     # mm: narrow strips round a known axis with corners this near one radius ...
LAND_MIN = 3         # ... at least this many of them (and MIN_SPAN_DEG between them) ...
LAND_MAX_FACETS = 12 # ... of at most this many facets each (bigger ones stand alone) ...
LAND_TILT_DEG = 3    # ... each facet facing this near straight out from the axis: cylinders
CREASE_DEG = 20      # a one-ring cone must meet a neighbour at a crease at least this sharp
MIN_CORNERS = {"cylinder": 8, "cone": 10, "sphere": 8, "torus": 12}
# A patch whose outline doesn't follow its own boundary lines (e.g. a hole running
# out through a sloped face, or a rounded knob cut by another feature) is accepted as
# a "trimmed" patch. Without the outline check to lean on, it needs more evidence:
WHOLE_ON_SURFACE = 0.6     # a region fitted whole: this share of its corners on the surface
WHOLE_STRAY = 0.02          # mm: how far the rest (outline corners on a seam) may stray
# A region taken whole although its corners stray from the surface (a mesh exported
# coarsely, or rounded off, keeps its cylinders only to a hundredth of a millimetre or
# so; cut into strips instead, every strip fits a cylinder of its own, wrong radius):
NOISY_MAX = 0.02 * ACCURACY  # mm: every corner this close to the surface ...
NOISY_TYPICAL = 0.01 * ACCURACY  # ... nearly all of them (95%) this close ...
NOISY_TURN_DEG = 60         # ... the region turning at least this far (a gently curved patch fits anything)
NOISY_MIN_FACETS = 12       # ... with at least this many facets
TRIMMED_MIN_CORNERS = 12
TRIMMED_MIN_FACETS = 6
TRIMMED_MIN_TURN_DEG = 15   # the facets must face in directions at least this far apart

# Loose pass (run last, on leftovers only; the approach of the stlToSolid project):
# accept any patch the surface explains geometrically, without the outline and crease
# rules. Accuracy is still guaranteed by the corner, interior and normal checks; the
# strict passes run first so clean patches keep their exact natural boundaries.
LOOSE_NORMAL_DEG = 8        # facet normal vs surface normal (plus the facet's own spread)
LOOSE_MAX_GAP = 0.08        # mm: facet interiors (centres, edge midpoints) vs surface
LOOSE_MIN_TURN_DEG = {"cylinder": 6, "cone": 6, "sphere": 20, "torus": 20}
LOOSE_MIN_FACETS = {"cylinder": 3, "cone": 6, "sphere": 8, "torus": 8}
_loose = False
_anchored = False   # the surface is pinned by the faces around it (fillets.py): trust it
_axis_vouched = False   # on an axis another patch already has, in a rounded-off mesh: a cut-to-
                        # shape patch may hand over smoothly (to its own fillets) all round
TWO_PI = 2 * math.pi
_deadline = None            # time.time() by which the whole conversion should be done
_budget = None              # (seconds; both set by stl2curves.stl_to_solid)


def time_left():
    """Seconds left of the conversion's time limit (infinite without one). Only the
    optional refinements heed it (fitting blends, hunting down troublemakers): finding
    patches cut short leaves the rest to become hundreds of slow blends."""
    return math.inf if _deadline is None else _deadline - time.time()


# ---------------------------------------------------------------- mesh

def load_stl(path):
    """Return welded vertex array and triangle index array for a binary or ASCII STL (or
    an OBJ, PLY or OFF file, by its extension).

    The file is read by MMM's reader (mmm/meshio.py), which takes the same numbers from
    a well-formed file and refuses a broken one with a message for the user (MeshError):
    our own parse took only the exact binary layout and otherwise any line starting with
    "vertex", and crashed on empty files, wrong counts and stray tokens. Triangles with
    broken numbers are dropped, and a mesh with nothing left after welding is refused."""
    from mmm import meshio as mio
    ext = Path(path).suffix.lower()
    tris = mio.read_any(str(path)) if ext in (".obj", ".ply", ".off") else mio.read_stl(str(path))
    tris = tris[np.isfinite(tris).all(axis=(1, 2))]
    verts = tris.reshape(-1, 3)
    if len(verts):
        diag = float(np.linalg.norm(verts.max(0) - verts.min(0)))
        if diag > 0:
            mio.check_size(diag, float(np.abs(verts).max()))
            # (the weld's integer keys below overflow int64 past 9e14 mm)
            if float(np.abs(verts).max()) > 9e13:
                raise mio.MeshError("This model's coordinates are too large for stl2curves' reader. "
                                    "Scale it down in the program that made it (it may have been saved "
                                    "in the wrong unit), and export it again.")
    # corners joined on a WELD_GRID mm grid, which closes the hairline cracks of files that
    # write a shared corner twice slightly differently (see exact_weld for very fine meshes)
    pts, tris = _welded(verts, np.round(verts * (1.0 / WELD_GRID)).astype(np.int64))
    if not len(tris):
        raise mio.MeshError("Every triangle in this mesh is squashed flat (or has broken numbers), so there is "
                            "no shape to convert.")
    return pts, tris


def exact_weld(path):
    """(points, triangles) with only identical corners joined, or None unless that leaves a
    clean closed mesh (no open edge, no edge of three or more triangles). For a mesh finer
    than WELD_GRID, whose grid weld joins the ends of real edges: Bravo's motor shaft joint
    (469k triangles across 27 mm, 920 edges under 1e-4 mm) came out of the grid with 42
    edges of three triangles, 24 still there after repair, so no solid could be closed and
    the build fell back to 25,488 patches, though the file is watertight as written
    (2026-10-30). Used only where the repaired grid weld still has such edges (merged.py)."""
    from mmm import meshio as mio
    ext = Path(path).suffix.lower()
    tris = mio.read_any(str(path)) if ext in (".obj", ".ply", ".off") else mio.read_stl(str(path))
    tris = tris[np.isfinite(tris).all(axis=(1, 2))]
    verts = tris.reshape(-1, 3)
    # (identical as stored: the 32-bit floats' bits, with -0 and +0 made one)
    pts, tris = _welded(verts, (verts.astype(np.float32) + np.float32(0)).view(np.int32).astype(np.int64))
    return (pts, tris) if len(tris) and not _defects(tris) else None


def _welded(verts, key):
    """(points, triangles): the corners with equal keys joined into one, triangles left
    without area dropped."""
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    pts = np.zeros((len(uniq), 3))
    pts[inv.ravel()] = verts
    tris = inv.reshape(-1, 3)
    return pts, tris[(tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 2] != tris[:, 0])]


def _defects(tris):
    """How many edges are open (one triangle) or shared by three or more triangles."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, count = np.unique(e, axis=0, return_counts=True)
    return int((count == 1).sum() + (count > 2).sum())


def _labels(n, a, b):
    """Connected-component label for each of n items linked by pairs (a[i], b[i])."""
    parent = np.arange(n)

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(a, b):
        ri, rj = root(i), root(j)
        if ri != rj:
            parent[ri] = rj
    roots = np.array([root(i) for i in range(n)])
    return np.unique(roots, return_inverse=True)[1]


def _groups(labels):
    order = np.argsort(labels, kind="stable")
    return np.split(order, np.nonzero(np.diff(labels[order]))[0] + 1)


def _edges(tris):
    return np.sort(tris[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)


def _edge_counts(tris, npts):
    """np.unique(_edges(tris), axis=0, return_counts=True), by one integer key per edge:
    the same rows in the same order (lowest corner, then highest), without sorting the
    rows as records (a good part of a big mesh's analysis)."""
    E = _edges(tris)
    m = np.int64(npts) + 1
    key, counts = np.unique(E[:, 0].astype(np.int64) * m + E[:, 1], return_counts=True)
    return np.stack([key // m, key % m], axis=1).astype(E.dtype), counts


def _uniq(a):
    """np.unique of a 1-d integer array (sorted, each value once), at a third of its
    cost on big arrays and half on the small ones analysis asks for by the million."""
    a = np.sort(a)
    if len(a) < 2:
        return a
    keep = np.empty(len(a), bool)
    keep[0] = True
    np.not_equal(a[1:], a[:-1], out=keep[1:])
    return a[keep]


MERGE_MAX_DEG = 1     # rounding noise never explains a bigger tilt than this
AXIS_NOISE = 10       # on a known axis, corners may sit this many rounding steps off
MIN_VOUCHED_CONE_DEG = 5    # ... but a cone found that way must taper at least this much
PAIR_BLOCK = 512      # rows: facet normals compared a block at a time (_least_pair_dot)
STRADDLE_STEPS = 5    # a leftover flat facet this many tolerances off a neighbouring patch's
                      # surface joins it (_absorb_straddlers)


def grid_noise(pts):
    """How far corners may sit off the true surface because the file rounded their
    coordinates to a grid (many exports write 0.001 mm steps): half a step along each
    axis. 0 when the coordinates aren't rounded."""
    for step in (0.01, 0.001, 0.0001):
        err = np.abs(pts - np.round(pts / step) * step)
        if (err <= 2.5e-7 * np.abs(pts) + 1e-7).mean() >= 0.75:
            return step * math.sqrt(3) / 2
    return 0.0


def _merge_flat(pts, tris, tn, tarea, labels, t1, t2, noise, tol):
    """Join neighbouring facets that differ only by rounding noise. Rounded corners tilt
    a long thin triangle by up to noise / its width, which splits one flat face into many
    facets. Two triangles are joined when their tilt is within what the noise explains,
    and only while all the corners of the joined facet still lie on one plane (so a
    finely cut curve can't creep into a plane one strip at a time)."""
    if noise <= 0 or not len(t1):
        return labels
    corners = pts[tris]
    longest = np.linalg.norm(corners - corners[:, [1, 2, 0]], axis=2).max(axis=1)
    width = 2 * tarea / np.maximum(longest, 1e-12)
    dot = np.clip(np.einsum("ij,ij->i", tn[t1], tn[t2]), -1, 1)
    allowed = np.minimum(noise / np.maximum(width[t1], 1e-12) + noise / np.maximum(width[t2], 1e-12),
                         math.radians(MERGE_MAX_DEG))    # (specks narrower than the noise: no)
    ok = (np.arccos(dot) <= allowed) & (labels[t1] != labels[t2])
    if not ok.any():
        return labels
    verts = {}
    for t, f in enumerate(labels):
        verts.setdefault(int(f), set()).update(tris[t].tolist())
    parent = list(range(labels.max() + 1))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in np.nonzero(ok)[0][np.argsort(-dot[ok])]:
        x, y = root(int(labels[t1[i]])), root(int(labels[t2[i]]))
        if x == y:
            continue
        both = verts[x] | verts[y]
        P = pts[list(both)]
        mid = P.mean(axis=0)
        n = np.linalg.svd(P - mid, full_matrices=False)[2][2]
        if np.abs((P - mid) @ n).max() <= tol + noise:
            parent[x] = y
            verts[y] = both
            del verts[x]
    return np.unique([root(int(f)) for f in labels], return_inverse=True)[1]


# what a conversion keeps on a Mesh for itself, left out when it is pickled
# (Mesh.__getstate__): the build's caches, and the file the mesh was shared with the
# workers in, which is this run's own (a saved stage taken up again sent the workers
# looking for a file long gone, and the pool was taken for broken)
BUILD_ONLY = ("face_cache", "face_cache_features", "corner_pool", "blend_cache", "fills", "fill_jobs",
              "bare_ahead", "file_dropped", "shared_as")


class _Packed:
    """A list of small arrays, of lists of numbers, or a dict of number pairs to such
    lists, held as a few flat arrays for pickling (see Mesh.__getstate__); unpack() gives
    back the same values, of the same types, in the same order."""

    def __init__(self, kind, *parts):
        self.kind, self.parts = kind, parts

    @staticmethod
    def _offsets(lengths):
        return np.cumsum(np.asarray(lengths, np.int64))

    @classmethod
    def arrays(cls, items):
        if not items:
            return items
        return cls("arrays", np.concatenate(items), cls._offsets([len(a) for a in items])[:-1])

    @classmethod
    def lists(cls, items, dtype):
        flat = np.fromiter((x for l in items for x in l), dtype, sum(len(l) for l in items))
        return cls("lists", flat, cls._offsets([len(l) for l in items]))

    @classmethod
    def table(cls, d):
        keys = np.array(list(d.keys())) if d else np.zeros((0, 2), np.int64)
        values = list(d.values())
        lengths = [len(v) for v in values]
        flat = [x for v in values for x in v]
        return cls("table", keys, cls._offsets(lengths), np.array(flat) if flat else np.zeros(0, np.int64))

    def unpack(self):
        if self.kind == "arrays":
            flat, cuts = self.parts
            return np.split(flat, cuts)
        if self.kind == "lists":
            flat, ends = self.parts
            starts = np.r_[0, ends[:-1]]
            return [list(flat[a:b]) for a, b in zip(starts.tolist(), ends.tolist())]
        keys, ends, flat = self.parts
        starts = np.r_[0, ends[:-1]]
        # (keys as pairs of numpy numbers and values as numpy numbers or rows, as built)
        return {(a, b): list(flat[lo:hi]) for (a, b), lo, hi in zip(keys, starts.tolist(), ends.tolist())}


class Mesh:
    """Triangles grouped into flat facets, with which facets meet at a gentle bend."""

    def __init__(self, pts, tris):
        a, b, c = (pts[tris[:, k]] for k in range(3))
        cross = np.cross(b - a, c - a)
        size = np.linalg.norm(cross, axis=1)
        keep = size > 1e-12
        self.pts, self.tris = pts, tris[keep]
        self.tn = cross[keep] / size[keep, None]
        self.tarea = size[keep] / 2

        edges = _edges(self.tris)
        owner = np.repeat(np.arange(len(self.tris)), 3)
        order = np.lexsort((edges[:, 1], edges[:, 0]))
        edges, owner = edges[order], owner[order]
        shared = np.nonzero(np.all(edges[1:] == edges[:-1], axis=1))[0]
        t1, t2 = owner[shared], owner[shared + 1]
        dot = np.einsum("ij,ij->i", self.tn[t1], self.tn[t2])
        flat = dot > math.cos(math.radians(COPLANAR_DEG))

        self.facet_of = _labels(len(self.tris), t1[flat], t2[flat])
        self.noise = grid_noise(pts)
        self.facet_of = _merge_flat(self.pts, self.tris, self.tn, self.tarea, self.facet_of,
                                    t1[~flat], t2[~flat], self.noise, TOL + FLOAT_TOL * float(np.abs(pts).max()))
        nf = self.facet_of.max() + 1
        self.ftris = _groups(self.facet_of)
        self.farea = np.bincount(self.facet_of, self.tarea, nf)
        fn = np.zeros((nf, 3))
        np.add.at(fn, self.facet_of, self.tn * self.tarea[:, None])
        self.fn = fn / np.linalg.norm(fn, axis=1)[:, None]
        fc = np.zeros((nf, 3))
        np.add.at(fc, self.facet_of, self.pts[self.tris].mean(axis=1) * self.tarea[:, None])
        self.fcent = fc / self.farea[:, None]
        self.fverts = [_uniq(self.tris[t].ravel()) for t in self.ftris]

        f1, f2 = self.facet_of[t1[~flat]], self.facet_of[t2[~flat]]
        ends = edges[shared[~flat]]
        pairs = np.sort(np.c_[f1, f2], axis=1)
        bend = np.einsum("ij,ij->i", self.fn[pairs[:, 0]], self.fn[pairs[:, 1]])
        smooth = (pairs[:, 0] != pairs[:, 1]) & (bend > math.cos(math.radians(SMOOTH_DEG)))
        self.edge_facets = {}   # (vertex, vertex), sorted -> facets using that edge
        for (a, b), f in zip(edges, self.facet_of[owner]):
            self.edge_facets.setdefault((a, b), []).append(f)
        self.nbrs = [set() for _ in range(nf)]
        self.shared = {}   # (facet, facet) -> list of shared edges (vertex id pairs)
        for (x, y), e in zip(pairs[smooth], ends[smooth]):
            self.nbrs[x].add(y)
            self.nbrs[y].add(x)
            self.shared.setdefault((x, y), []).append(e)

    def __getstate__(self):
        # (the neighbour sets as lists in the order they have: a set pickle rebuilds can
        # come out in another order, and the order decides what is tried, so a worker or
        # a saved stage taken up again went its own way)
        state = dict(self.__dict__)
        state["nbrs"] = _Packed.lists([list(s) for s in self.nbrs], self.facet_of.dtype)
        # (nor what the build keeps on the mesh: OpenCascade faces and corners, which
        # don't pickle, and work under way. Shared with the workers mid-build, the mesh
        # failed to pickle, the pool was taken for broken, and the rest of the part was
        # built here, every fill and cut in turn and without time limits)
        for key in BUILD_ONLY:
            state.pop(key, None)
        # The per-facet lists and the edge tables go as a few flat arrays, the tables to
        # be rebuilt only where they are asked for. Pickled as they are, as millions of
        # small objects (a numpy scalar or array is a few hundred bytes read back), a
        # 920k-triangle mesh came back 3.2 GB in each worker, where it had taken 1.4 GB
        # here, and 16 workers ran the machine out of memory; most workers never look at
        # the edge tables at all. Read back, everything is as it was: the same values,
        # types and order (the lists' arrays are views of one array, as ftris's are here).
        for key in ("fverts", "ftris"):
            state[key] = _Packed.arrays(state[key])
        lazy = dict(state.pop("_lazy", {}))
        for key in ("edge_facets", "shared"):
            if key in state:
                lazy[key] = _Packed.table(state.pop(key))
        state["_lazy"] = lazy
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        for key in ("fverts", "ftris", "nbrs"):
            if isinstance(self.__dict__.get(key), _Packed):
                self.__dict__[key] = self.__dict__[key].unpack()

    def __getattr__(self, name):
        # (an edge table read back from a pickle, rebuilt the first time it is asked for)
        lazy = self.__dict__.get("_lazy")
        if lazy and name in lazy:
            value = self.__dict__[name] = lazy.pop(name).unpack()
            return value
        raise AttributeError(name)

    def smooth_regions(self):
        nf = len(self.fn)
        a = [x for x in range(nf) for y in self.nbrs[x] if x < y]
        b = [y for x in range(nf) for y in self.nbrs[x] if x < y]
        return [g for g in _groups(_labels(nf, a, b)) if len(g) >= 3]

    def shared_direction(self, f, g):
        """Direction of the (longest) edge shared by two facets."""
        e = self.shared.get((min(f, g), max(f, g)))
        if not e:
            return None
        vec = [self.pts[j] - self.pts[i] for i, j in e]
        v = max(vec, key=np.linalg.norm)
        return v / np.linalg.norm(v)


# ---------------------------------------------------------------- surfaces

def _frame(d):
    """Unit axis d and two unit directions square to it (d, e1, e2 right-handed).
    Written out by hand: numpy's cross and norm cost more than the arithmetic on three
    numbers, and this runs millions of times."""
    x, y, z = float(d[0]), float(d[1]), float(d[2])
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    a, b, c = (0.0, z, -y) if abs(x) < 0.9 else (-z, 0.0, x)    # d x (1,0,0), d x (0,1,0)
    m = math.sqrt(a * a + b * b + c * c)
    a, b, c = a / m, b / m, c / m
    return (np.array([x, y, z]), np.array([a, b, c]),
            np.array([y * c - z * b, z * a - x * c, x * b - y * a]))


class Revolved:
    """Surface of revolution: cylinder, cone, sphere or torus.

    Around the axis (point a, direction d), a point has radius rho and height z.
    The profile is a line rho = c0 + k*z (cylinder when k == 0, else cone) or a
    circle of radius r centred at (rc, zc) (sphere when rc == 0, else torus).
    """

    def __init__(self, a, d, line=None, circle=None):
        self.d, self.e1, self.e2 = _frame(d)
        a = np.asarray(a, float)
        self.a = a - (a @ self.d) * self.d
        self.line, self.circle = line, circle
        if line:
            self.scale = max(abs(line[0]), 1.0)
        else:
            self.scale = circle[2]

    @property
    def kind(self):
        if self.line:
            return "cylinder" if self.line[1] == 0 else "cone"
        return "sphere" if self.circle[0] == 0 else "torus"

    def local(self, p):
        v = np.atleast_2d(p) - self.a
        z = v @ self.d
        w = v - np.outer(z, self.d)
        return np.linalg.norm(w, axis=1), z, w

    def angle(self, w):
        return np.arctan2(w @ self.e2, w @ self.e1)

    def signed(self, p):
        """Distance to the surface; positive = away from the axis (line) / tube centre (circle)."""
        rho, z, _ = self.local(p)
        if self.line:
            c0, k = self.line
            return (rho - c0 - k * z) / math.hypot(1, k)
        rc, zc, r = self.circle
        return np.hypot(rho - rc, z - zc) - r

    def normal(self, p):
        return self._normal(*self.local(p))

    def signed_normal(self, p):
        """signed(p) and normal(p) at once, the points placed round the axis only once
        (the same numbers: Region._test asks for both, millions of times)."""
        rho, z, w = self.local(p)
        if self.line:
            c0, k = self.line
            signed = (rho - c0 - k * z) / math.hypot(1, k)
        else:
            rc, zc, r = self.circle
            signed = np.hypot(rho - rc, z - zc) - r
        return signed, self._normal(rho, z, w)

    def _normal(self, rho, z, w):
        radial = w / np.maximum(rho, 1e-12)[:, None]
        radial[rho < 1e-9] = 0
        if self.line:
            c0, k = self.line
            nr = np.full(len(rho), 1 / math.hypot(1, k))
            nz = np.full(len(rho), -k / math.hypot(1, k))
        else:
            rc, zc, r = self.circle
            nr, nz = rho - rc, z - zc
            size = np.maximum(np.hypot(nr, nz), 1e-12)
            nr, nz = nr / size, nz / size
        return radial * nr[:, None] + np.outer(nz, self.d)

    def point(self, rho, z, u):
        return self.a + z * self.d + rho * (math.cos(u) * self.e1 + math.sin(u) * self.e2)


class Sphere:
    def __init__(self, c, r):
        self.c, self.r, self.scale = np.asarray(c, float), float(r), float(r)

    kind = "sphere"

    def signed(self, p):
        return np.linalg.norm(np.atleast_2d(p) - self.c, axis=1) - self.r

    def normal(self, p):
        v = np.atleast_2d(p) - self.c
        return v / np.maximum(np.linalg.norm(v, axis=1), 1e-12)[:, None]

    def signed_normal(self, p):
        """signed(p) and normal(p) at once (as Revolved.signed_normal)."""
        v = np.atleast_2d(p) - self.c
        d = np.linalg.norm(v, axis=1)
        return d - self.r, v / np.maximum(d, 1e-12)[:, None]


_mesh_tol = TOL


def _tol(model=None):
    return _mesh_tol


# ---------------------------------------------------------------- fitting

def _circle2d(x, y):
    if len(np.unique(np.round(np.c_[x, y], 5), axis=0)) < 3:
        return None
    sol = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], x * x + y * y, rcond=None)[0]
    cx, cy = sol[0] / 2, sol[1] / 2
    r2 = sol[2] + cx * cx + cy * cy
    return (cx, cy, math.sqrt(r2)) if r2 > 0 else None


def _fits(model, P):
    return model is not None and np.abs(model.signed(P)).max() <= _tol(model)


def _cylinder(d, P):
    d, e1, e2 = _frame(d)
    c = _circle2d(P @ e1, P @ e2)
    if c is None or c[2] > 1e4:
        return None
    return Revolved(c[0] * e1 + c[1] * e2, d, line=(c[2], 0))


def _cone(N, P):
    """Cone through facets whose normals N all make the same angle with the axis."""
    if len(np.unique(np.round(N, 4), axis=0)) < 3:
        return None
    Nc = N - N.mean(axis=0)
    d = np.linalg.eigh(Nc.T @ Nc)[1][:, 0]
    return _line_on_direction(d, P)


def _line_on_direction(d, P):
    d, e1, e2 = _frame(d)
    x, y, z = P @ e1, P @ e2, P @ d
    sol = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x), z, z * z], x * x + y * y, rcond=None)[0]
    cx, cy = sol[0], sol[1]
    rho = np.hypot(x - cx, y - cy)
    if np.ptp(z) < 1e-6:
        return None
    k, c0 = np.polyfit(z, rho, 1)
    if abs(k) > math.tan(math.radians(80)):
        return None                     # nearly flat: leave it as a plane
    if _straight(k, z):
        k, c0 = 0, rho.mean()
    return Revolved(cx * e1 + cy * e2, d, line=(c0, k))


def _straight(k, z):
    """Is a line profile's slope k (over heights z) no slope at all? (A rounded-off mesh
    tilts a cylinder's fit by a hair; straightened, it still fits within tolerance.)"""
    return abs(k) < 1e-4 or abs(k) * np.ptp(z) / 2 <= _tol()


def _rings_refit(model, P):
    """Refit a cylinder/cone from its circular vertex rings: much more precise than normals."""
    rho, z, _ = model.local(P)
    levels = np.round(z / 0.01)
    centers, radii, heights, normals = [], [], [], []
    for lv in np.unique(levels):
        ring = P[levels == lv]
        if len(ring) < 3:
            continue
        mid = ring.mean(axis=0)
        n = _right_vectors(ring - mid)[2]
        _, e1, e2 = _frame(n)
        c = _circle2d((ring - mid) @ e1, (ring - mid) @ e2)
        if c is None:
            continue
        centers.append(mid + c[0] * e1 + c[1] * e2)
        radii.append(c[2])
        heights.append(lv)
        normals.append(n if n @ model.d > 0 else -n)
    if len(centers) < 2:
        return None
    centers = np.array(centers)
    lo, hi = np.argmin(heights), np.argmax(heights)
    d = centers[hi] - centers[lo]
    if np.linalg.norm(d) < 1e-6:
        return None
    d /= np.linalg.norm(d)
    if model.line[1] == 0:
        return _cylinder(d, P)
    return _line_on_direction(d, P)


def _sphere(P):
    if len(P) < 4 or np.linalg.svd(P - P.mean(axis=0), compute_uv=False)[2] < 1e-6:
        return None
    sol = np.linalg.lstsq(np.c_[2 * P, np.ones(len(P))], (P * P).sum(axis=1), rcond=None)[0]
    r2 = sol[3] + sol[:3] @ sol[:3]
    return Sphere(sol[:3], math.sqrt(r2)) if 0 < r2 < 1e8 else None


def _on_axis(axis, P):
    """Cone/cylinder (line profile) or torus/sphere (circle profile) around a known axis."""
    base = Revolved(axis[0], axis[1], line=(1.0, 0))
    rho, z, _ = base.local(P)
    if len(np.unique(np.round(np.c_[rho, z], 5), axis=0)) < 3:
        return None
    if np.ptp(z) > 1e-6:
        k, c0 = np.polyfit(z, rho, 1)
        if _straight(k, z):
            k, c0 = 0, rho.mean()
        line = Revolved(axis[0], axis[1], line=(c0, k))
        if abs(k) < math.tan(math.radians(80)) and _fits(line, P):
            return line
    c = _circle2d(rho, z)
    if c is None or c[2] > 1e4:
        return None
    rc = 0.0 if abs(c[0]) < TOL else c[0]
    return Revolved(axis[0], axis[1], circle=(rc, c[1], c[2]))


# ---------------------------------------------------------------- growing patches

class Region:
    """A set of smoothly connected facets, with fast "which facets lie on this surface" tests."""

    def __init__(self, mesh, facets):
        self.mesh = mesh
        self.facets = np.asarray(facets)
        pos = {f: i for i, f in enumerate(self.facets)}
        lists = [mesh.fverts[f] for f in self.facets]
        lens = np.array([len(l) for l in lists])
        self.vids, self.vinv = np.unique(np.concatenate(lists), return_inverse=True)
        self.lens = lens
        self.starts = np.r_[0, np.cumsum(lens)[:-1]]
        self.fv = np.split(self.vinv, np.cumsum(lens)[:-1])
        self.P = mesh.pts[self.vids]
        self.n = mesh.fn[self.facets]
        self.c = mesh.fcent[self.facets]
        self.nbrs = [[pos[g] for g in mesh.nbrs[f] if g in pos] for f in self.facets]
        self.size = np.array([np.ptp(self.P[v], axis=0).max() for v in self.fv])
        # Needle-thin facets have unreliable plane normals even when their corners are exact.
        self.sliver = mesh.farea[self.facets] < 0.05 * self.size ** 2
        self.free = np.ones(len(self.facets), bool)
        self.explored = np.zeros(len(self.facets), bool)
        self.reads = self.marks = None  # (while trying pairs ahead in a worker: see _seed_slice)

    def points(self, idx):
        return self.P[_uniq(np.concatenate([self.fv[i] for i in idx]))]

    def _corners(self, idx):
        """The corners of facets idx (positions in self.P), facet after facet, where each
        facet's corners begin among them, and how many each has. Gathered in one go: a
        list of each facet's corner array, joined, cost more than the tests on them."""
        lens = self.lens[idx]
        ends = np.cumsum(lens)
        heads = ends - lens
        return self.vinv[np.repeat(self.starts[idx] - heads, lens) + np.arange(ends[-1])], heads, lens

    def _test(self, model, idx):
        """For facets idx: does each lie on the surface, which way does it face, surface normals."""
        idx = np.asarray(idx, dtype=np.intp)
        verts, heads, lens = self._corners(idx)
        # (distance and normal together where both are wanted: a few microseconds a
        # call, but grow asks millions of times)
        both = getattr(model, "signed_normal", None)
        if both is not None:
            centre_d, surface_n = both(self.c[idx])
        else:
            centre_d, surface_n = model.signed(self.c[idx]), model.normal(self.c[idx])
        if _loose and both is not None:
            corner_d, corner_n = both(self.P[verts])
        else:
            corner_d = model.signed(self.P[verts])
        worst = np.maximum.reduceat(np.abs(corner_d), heads)
        facing = np.einsum("ij,ij->i", self.n[idx], surface_n)
        if _loose:
            # the surface normal's own spread over the facet is allowed on top: the
            # widest angle between the surface normals at its corners, worked out for
            # all the facets at once (facets of each corner count together; one facet
            # at a time it was most of the loose pass on a 900k-triangle mesh)
            if both is None:
                corner_n = model.normal(self.P[verts])
            if (lens == lens[0]).all():
                S = corner_n.reshape(len(idx), int(lens[0]), 3)
                least = (S @ S.transpose(0, 2, 1)).min(axis=(1, 2))
            else:
                least = np.empty(len(idx))
                for count in np.unique(lens).tolist():
                    at = np.flatnonzero(lens == count)
                    S = corner_n[(heads[at, None] + np.arange(count)).ravel()].reshape(len(at), count, 3)
                    least[at] = (S @ S.transpose(0, 2, 1)).min(axis=(1, 2))
            spread = np.array([math.acos(max(-1.0, min(1.0, x))) for x in least.tolist()])
            aligned = np.abs(facing) >= np.cos(np.minimum(np.radians(LOOSE_NORMAL_DEG) + spread,
                                                          np.radians(60)))
            gap = LOOSE_MAX_GAP
        else:
            aligned = (np.abs(facing) >= math.cos(math.radians(NORMAL_DEG))) | self.sliver[idx]
            gap = MAX_SAG
        on = (worst <= _tol(model)) & (np.abs(centre_d) <= gap) & aligned
        if self.reads is not None:
            # (see _seed_slice: only the facets whose answer turned on whether they were
            # free. One off the surface is refused either way; counted, every facet
            # round a patch's edge, often a neighbouring patch's, made the replay redo
            # the try here once that neighbour was found: a 900k-triangle mesh redid
            # most of its loose pass in this process, the workers idle)
            self.reads.append(idx[on])
        ok = on & self.free[idx]
        return ok, facing > 0, surface_n

    def grow(self, model, seeds):
        """Facets connected to the seeds that lie on the model surface, and whether it's convex.

        A facet only joins if it bends away from its neighbour the way the surface
        does. (A flat wall beside a rounded edge can have its corners on some huge
        circle, but it meets the rounded edge without the bend that circle predicts.)
        """
        seeds = list(seeds)
        ok, out, sn = self._test(model, seeds)
        if not ok.all() or (out != out[0]).any():
            return None, None
        convex = bool(out[0])
        slack = math.radians(BEND_SLACK_DEG)
        small = 0.1 * model.scale
        normal = dict(zip(seeds, sn))

        def bends_right(i, j, nj):
            if (self.size[i] < small and self.size[j] < small) or self.sliver[i] or self.sliver[j]:
                return True             # tiny or needle-thin facets: the corner check is what counts
            actual = math.acos(min(1.0, self.n[i] @ self.n[j]))
            predicted = math.acos(min(1.0, normal[i] @ nj))
            return abs(actual - predicted) <= slack

        for a in seeds[1:]:
            if not bends_right(seeds[0], a, normal[a]):
                return None, None
        # A facet that fails only the bend test is looked at again each time another
        # of its neighbours joins, since the verdict depends on which neighbour it's
        # compared with.
        tested, frontier = {}, seeds
        while frontier:
            cand = sorted({j for i in frontier for j in self.nbrs[i]
                           if j not in normal and tested.get(j, (True,))[0]})
            new = [j for j in cand if j not in tested]
            if new:
                tested.update(zip(new, zip(*self._test(model, new))))
            frontier = []
            for j in cand:
                good, o, nj = tested[j]
                if good and o == convex and any(i in normal and bends_right(i, j, nj)
                                                 for i in self.nbrs[j]):
                    normal[j] = nj
                    frontier.append(j)
        return np.array(sorted(normal)), convex


def _refit(model, P, axis_fixed):
    if isinstance(model, Sphere):
        return _sphere(P)
    if axis_fixed:
        return _on_axis((model.a, model.d), P)
    if model.line:
        return _rings_refit(model, P)
    return None


def _grow_refit(region, model, seeds, axis_fixed):
    idx, convex = region.grow(model, seeds)
    if idx is None:
        return None
    for _ in range(8):  # each refit uses more of the surface, so it can reach a little further
        better = _refit(model, region.points(idx), axis_fixed)
        if better is None or (not axis_fixed and better.kind != model.kind):
            break
        idx2, convex2 = region.grow(better, seeds)
        if idx2 is None or len(idx2) < len(idx):
            break
        grew = len(idx2) > len(idx)
        model, idx, convex = better, idx2, convex2
        if not grew:
            break
    return model, idx, convex


def _profile_param(model, P):
    """Position of points along the profile: height for a line, angle round the tube for a circle."""
    rho, z, _ = model.local(P)
    if model.line:
        return z, 1.0
    rc, zc, r = model.circle
    return np.arctan2(z - zc, rho - rc), r


def _trim(region, model, idx):
    """Drop facets that leaked past a patch's true end circles.

    Where two surfaces meet tangentially, the first row of tiny facets on the
    neighbour can sit within tolerance of this surface too. A real end circle
    carries many mesh corners; a leaked sliver adds only a few stray ones, so the
    ends are taken as the outermost well-populated levels.
    """
    tol = _tol(model)
    for _ in range(3):
        verts = [region.fv[i] for i in idx]
        used = np.unique(np.concatenate(verts))
        # (only the patch's own corners placed: the whole region's took most of the
        # loose pass's time on a 900k-triangle mesh, a region being most of the part)
        t = np.empty(len(region.P))
        t[used], scale = _profile_param(model, region.P[used])
        levels, counts = np.unique(np.round(t[used] * scale / tol), return_counts=True)
        busy = levels[counts >= max(4, 0.2 * counts.max())] * tol / scale
        if len(busy) == 0:
            break
        lo, hi = busy.min() - 2 * tol / scale, busy.max() + 2 * tol / scale
        keep = np.array([((t[v] >= lo) & (t[v] <= hi)).all() for v in verts])
        if keep.all():
            break
        idx = idx[keep]
        if len(idx) < 3:
            break
    return idx


def _candidate(mesh, region, model, seeds, axis_fixed=False):
    """Grow a hypothesis into a patch; return (Feature, facet positions, area) or None.

    Facets of a patch that grew but was rejected go into region.explored, so the
    same surface isn't regrown (and rejected again) from each of its facets.
    """
    if model is None:
        return None
    grown = _grow_refit(region, model, seeds, axis_fixed)
    if grown is None:
        return None
    model, idx, convex = grown
    options = []
    if isinstance(model, Revolved) and not (model.circle and model.circle[0] == 0):
        cut = _trim(region, model, idx)
        if len(cut) >= 3 and len(cut) < len(idx):
            better = _refit(model, region.points(cut), axis_fixed)
            if (better is not None and (axis_fixed or better.kind == model.kind)
                    and region._test(better, cut)[0].all()):
                options.append((better, cut))
            else:
                options.append((model, cut))
    options.append((model, idx))
    result = None
    min_facets = 2 if _loose else 3
    for m, ids in options:
        if len(ids) < min_facets or _fewer_distinct(np.round(region.n[ids], 3), min_facets):
            continue
        feature = _feature(mesh, m, region.facets[ids], convex)
        if feature is None:
            continue
        found = (feature, ids, float(mesh.farea[region.facets[ids]].sum()))
        if feature.kind != "trimmed":
            return found                # a clean outline: take it
        result = result or found
    if result is None and len(idx) >= 6:
        region.explored[idx] = True
        if region.marks is not None:
            region.marks.append(idx)                # (see _seed_slice)
    return result


def _fewer_distinct(rows, k):
    """Whether rows has fewer than k distinct rows, as np.unique(rows, axis=0) counts
    them (which sorts them all as records: most of the loose pass's time on a big mesh)."""
    rest = rows
    for _ in range(k):
        if not len(rest):
            return True
        rest = rest[1:][(rest[1:] != rest[0]).any(axis=1)]
    return False


def _right_vectors(A):
    """np.linalg.svd(A)[2], the right singular vectors, bit for bit, without the left
    ones: for a tall A numpy's default works out all M by M of those only for them to be
    thrown away (an outline of 10,000 points: an 800 MB matrix and half a second, for
    every facet absorbed into a big patch). LAPACK takes both from the same QR factor."""
    A = np.asarray(A)
    return np.linalg.svd(A, full_matrices=A.shape[0] < A.shape[1])[2]


def _least_pair_dot(n, near):
    """The smallest dot product between any two rows of n (unit facet normals),
    (n @ n.T).min(), as the caller needs it: compared with `near`. A small n gets exactly
    that. A big one gets a value
    on the same side of `near` as it, without the rows-by-rows matrix (a 16,570-facet
    patch's was 2.2 GB, built for every facet absorbed into it: a 920k-triangle mesh
    spent the best part of an hour so in _absorb_straddlers): a pair of rows already
    below `near` (the row farthest from their mean, and the row farthest from that one),
    or a bound above it (every row within some angle of their mean, no two are more
    than twice that apart), or failing both the minimum a block of rows at a time;
    within a hair of `near` the whole matrix after all, so every decision is as before."""
    if len(n) <= PAIR_BLOCK:
        return (n @ n.T).min()
    mean = n.sum(axis=0)
    a = int(np.argmin(n @ mean))
    dots = n @ n[a]
    pair = float(dots.min())
    if pair < near - 1e-9:
        return pair
    size = float(np.linalg.norm(mean))
    if size > 0:
        closest = float((n @ (mean / size)).min())
        if closest > 0 and 2 * closest * closest - 1 > near + 1e-9:
            return 2 * closest * closest - 1
    least = min(float((n[i:i + PAIR_BLOCK] @ n.T).min()) for i in range(0, len(n), PAIR_BLOCK))
    if abs(least - near) <= 1e-9:
        return (n @ n.T).min()
    return least


def _best(candidates):
    """The candidate covering the most area; among those covering about as much, the one
    fitting its facets far more closely (two rings of points lie on a cone and on a
    sphere alike; only the true surface also runs close between them)."""
    found = [c for c in candidates if c]
    if not found:
        return None
    best = max(found, key=lambda c: c[2])
    close = [c for c in found if c[2] >= 0.99 * best[2]]
    tight = min(close, key=lambda c: c[0].worst)
    return tight if tight[0].worst < 0.5 * best[0].worst else best


def _seed_candidates(mesh, region, i, j):
    """Every surface the facet pair (i, j) could belong to, grown as far as it goes."""
    P = region.points([i, j])
    near = [i, j] + [k for k in region.nbrs[i] + region.nbrs[j] if region.free[k]]
    near_pts = region.points(near)
    found = []
    # The cylinder's axis: along the edge the two facets share (a strip cut straight
    # across), or square to both their normals (every facet of a cylinder faces straight
    # out from the axis, however the strip is cut into triangles).
    axes = [mesh.shared_direction(region.facets[i], region.facets[j])]
    cross = np.cross(region.n[i], region.n[j])
    if np.linalg.norm(cross) > math.sin(math.radians(1)):
        cross /= np.linalg.norm(cross)
        if axes[0] is None or abs(cross @ axes[0]) < math.cos(math.radians(0.5)):
            axes.append(cross)
    for d in axes:
        if d is None:
            continue
        cyl = _cylinder(d, P)
        if _fits(cyl, P):
            found.append(_candidate(mesh, region, cyl, [i, j]))
    for k in region.nbrs[j]:
        if k != i and region.free[k]:
            pts = region.points([i, j, k])
            cone = _cone(region.n[[i, j, k]], pts)
            if _fits(cone, pts):
                found.append(_candidate(mesh, region, cone, [i, j, k]))
            break
    sv = np.linalg.svd(region.n[near], compute_uv=False) if len(near) >= 3 else None
    if not any(found) and sv is not None and sv[2] <= 0.1 * sv[0]:
        # Irregular triangles (a round whose edges run at different heights) fool both
        # quick guesses: estimate the axis from the whole neighbourhood's normals and
        # let a least-squares fit through the corners settle it. (Only where the normals
        # all lie square to one direction, as on a cylinder; not on doubly curved areas.)
        # (the loose pass meets most of the same neighbourhoods again: fitted once)
        fits = region.__dict__.setdefault("tube_fits", {})
        if tuple(near) not in fits:
            axis = _right_vectors(region.n[near])[2]
            cyl = _cylinder(axis, near_pts)
            fits[tuple(near)] = None if cyl is None else _tube_fit(near_pts, cyl)
        cyl = copy.deepcopy(fits[tuple(near)])      # (a model may be changed later on)
        if cyl is not None:
            if _fits(cyl, near_pts) and cyl.line[0] < 1e3:
                found.append(_candidate(mesh, region, cyl, [i, j]))
    # The same axis may carry a torus or sphere that explains more than one ring of facets.
    for c in [c for c in found if c]:
        ring = _on_axis((c[0].model.a, c[0].model.d), near_pts)
        if _fits(ring, near_pts) and ring.circle:
            found.append(_candidate(mesh, region, ring, [i, j], axis_fixed=True))
    for pts in (near_pts, P):
        ball = _sphere(pts)
        if _fits(ball, pts):
            found.append(_candidate(mesh, region, ball, [i, j]))
            break
    return _best(found)


def _whole_region(mesh, region):
    """The one cylinder, cone or sphere a whole smooth region lies on, if it is all one
    surface (a countersink, a plain hole or boss between two creases), or the torus a
    closed one does (a ring on its own: _ring). Fitted from every
    facet at once: a cone cut into long thin triangles fools any guess from a few facets
    (a thin strip of cone looks like a tilted cylinder)."""
    idx = np.arange(len(region.facets))
    P, N = region.points(idx), region.n
    if len(P) < MIN_CORNERS["cylinder"]:
        return None
    guesses = []
    sv, V = np.linalg.svd(N, full_matrices=False)[1:]
    if sv[2] <= 0.1 * sv[0]:
        guesses.append(_cylinder(V[2], P))      # every normal square to the axis
    guesses.append(_cone(N, P))                  # every normal at one angle to the axis
    found = []
    for model in guesses:
        if model is None:
            continue
        model = _tube_fit(P, model)
        if _fits(model, P):
            seed = [int(np.argmax(mesh.farea[region.facets]))]
            found.append(_candidate(mesh, region, model, seed))
        elif model is not None:
            found.append(_whole_despite_seams(mesh, region, model))
    ball = _sphere(P)
    if _fits(ball, P):
        found.append(_candidate(mesh, region, ball, [0]))
    elif not any(found):
        ring = _ring(mesh, region, P)
        if ring is not None:
            found.append(_candidate(mesh, region, ring, [0]))
    if not any(found):
        # (only cylinders and cones: their straight lines pin them down where a mesh
        # has strayed; a sphere would fit any gently rounded patch as well)
        found = [_whole_noisy(mesh, region, _tube_fit(P, m)) for m in guesses if m is not None]
    return _best(found)


def _ring(mesh, region, P):
    """The torus a whole closed smooth region (no outline: a ring on its own) lies on, or
    None. Its axis is the direction the corners spread least along, through their centre;
    the tube's centre circle and radius from their distances to it; then fitted to all of
    them (_tube_fit). Only for closed regions: a torus with an outline is found piece by
    piece round its axis, as before."""
    T = mesh.tris[np.concatenate([mesh.ftris[f] for f in region.facets])]
    _, counts = _edge_counts(T, len(mesh.pts))
    if len(P) < MIN_CORNERS["torus"] or (counts != 2).any():
        return None
    c = P.mean(axis=0)
    d = np.linalg.svd(P - c, full_matrices=False)[2][2]
    v = P - c
    z = v @ d
    rho = np.linalg.norm(v - np.outer(z, d), axis=1)
    rc = float(rho.mean())
    r = float(np.sqrt(np.mean((rho - rc) ** 2 + z ** 2)))
    if not r or rc <= r:
        return None
    model = _tube_fit(P, Revolved(c, d, circle=(rc, float(c @ _frame(d)[0]), r)))
    return model if model is not None and model.circle and _fits(model, P) else None


def _whole_noisy(mesh, region, model):
    """The whole region on this surface although its corners stray from it, by up to
    NOISY_MAX: a big region, curving far, every facet facing the surface's way."""
    if model is None or len(region.facets) < NOISY_MIN_FACETS:
        return None
    N = region.n
    mean = N.mean(axis=0)
    if np.linalg.norm(mean) > 0.5:
        turn = 2 * math.degrees(math.acos(max(-1.0, min(1.0, float((N @ (mean / np.linalg.norm(mean))).min())))))
        if turn < NOISY_TURN_DEG:
            return None
    off = np.abs(model.signed(region.P))
    if off.max() > NOISY_MAX or np.percentile(off, 95) > NOISY_TYPICAL:
        return None
    idx = np.arange(len(region.facets))
    ok, out, surface_n = region._test(model, idx)
    facing = np.abs(np.einsum("ij,ij->i", region.n, surface_n))
    if (out != out[0]).any() or (facing < math.cos(math.radians(NORMAL_DEG))).any():
        return None
    if (np.abs(model.signed(region.c)) > MAX_SAG).any():
        # a flat across the round (a D-shaped hole's: its corners on the circle, its
        # middle 1.15 mm inside it) is a face of its own, not a coarse facet of the round;
        # taken in, the cylinder couldn't be built round it and the hole was left faceted
        return None
    feature = _feature(mesh, model, region.facets, bool(out[0]))
    if feature is None:
        return None
    return feature, idx, float(mesh.farea[region.facets].sum())


def _whole_despite_seams(mesh, region, model):
    """The whole region on this surface although a few corners are off it: meshes where
    one face's triangulation put extra points on the straight chords of another face's
    outline (a few microns inside the curve). Accepted only if nearly every corner lies
    on the surface and the stray ones are all on the region's outline, close by."""
    idx = np.arange(len(region.facets))
    P = region.P
    for _ in range(4):
        off = np.abs(model.signed(P))
        keep = off <= max(_tol(model), 3 * np.median(off))
        if keep.mean() < 0.5 or keep.sum() < MIN_CORNERS[model.kind]:
            return None
        model = _tube_fit(P[keep], model)
        if model is None:
            return None
    off = np.abs(model.signed(P))
    on = off <= _tol(model)
    if on.mean() < WHOLE_ON_SURFACE or off.max() > WHOLE_STRAY:
        return None
    T = mesh.tris[np.concatenate([mesh.ftris[f] for f in region.facets])]
    uniq, counts = _edge_counts(T, len(mesh.pts))
    if not set(region.vids[~on].tolist()) <= set(uniq[counts == 1].ravel().tolist()):
        return None
    ok, out, surface_n = region._test(model, idx)
    facing = np.abs(np.einsum("ij,ij->i", region.n, surface_n))
    if (out != out[0]).any() or (facing < math.cos(math.radians(NORMAL_DEG))).any():
        return None
    feature = _feature(mesh, model, region.facets, bool(out[0]))
    if feature is None:
        return None
    return feature, idx, float(mesh.farea[region.facets].sum())


def find_features(pts, tris):
    """Return a Feature for every curved patch that can be rebuilt exactly."""
    return analyze(pts, tris)[1]


def analyze(pts, tris):
    """Return (mesh, features, tolerance): the facet structure and every rebuildable patch."""
    global _mesh_tol
    progress.step("grouping triangles into flat facets")
    mesh = Mesh(pts, tris)
    _mesh_tol = TOL + FLOAT_TOL * float(np.abs(pts).max())
    # (a part symmetric about a plane, wholly or in part, is searched on one side: what
    # is found there is tried mirrored on the other, see symmetry.py)
    mesh.mirror = mesh.mirror_skip = None
    if MIRROR:
        from . import symmetry
        progress.step("looking for mirror symmetry")
        found = symmetry.planes(mesh.pts, mesh.tris)
        if found:
            mesh.mirror = found[0][:2]
            mesh.mirror_skip = symmetry.mirrored(mesh, mesh.mirror)
    regions = [Region(mesh, g) for g in mesh.smooth_regions()]
    if mesh.mirror_skip is not None:
        # (only smooth areas lying wholly on the far side, each a twin of one on the near
        # side: one crossing the plane is searched as before, from every seed, or another
        # seed's guess wins there. Skipping the far half of a ring's dome let a cone that
        # fits it loosely be found first)
        for region in regions:
            if not mesh.mirror_skip[region.facets].all():
                mesh.mirror_skip[region.facets] = False
        if not mesh.mirror_skip.any():
            mesh.mirror = mesh.mirror_skip = None
    features = []

    # Pass 0: smooth regions that are all one surface, fitted whole.
    progress.step("fitting whole smooth areas")
    for region in regions:
        if len(region.facets) >= 6:
            best = _whole_region(mesh, region)
            if best:
                feature, idx, _ = best
                region.free[idx] = False
                features.append(feature)

    # Screw threads, among the curved facets still unexplained: before any pass that
    # tries surfaces round known axes (a thread's crest, root and flanks lie close to a
    # cylinder or cone on its own axis, which is often a hole's too) or the loose pass
    # (which would take pieces of them for cylinders and cones)
    from . import threads
    progress.step("looking for screw threads")
    free = [region.facets[region.free] for region in regions]
    if mesh.mirror_skip is not None:
        # (a facet mirroring onto the mesh is no thread's: mirrored, a thread turns the
        # other way and lands off the mesh)
        free = [f[~mesh.mirror_skip[f]] for f in free]
    screws = []
    for thread, fids in threads.find(mesh, np.concatenate(free) if free else np.zeros(0, int)):
        screws += threads.patches(mesh, thread, fids)
    claimed = [int(f) for x in screws for f in x.facets]
    for region in regions:
        region.free &= ~np.isin(region.facets, claimed)
    # a thread usually ends in a countersink or chamfer round its own axis, and runs on
    # into a plain bore: tried on that axis before pass 1 can cut them into strips that
    # each fit a cylinder of their own. Held to the thread's own tolerance: the CAD
    # program drew them together, both a hundredth or so off true.
    axes = [(t.a, t.d, 2 * float(t.profile[:, 1].max()) + 3) for t in {id(p.model.thread): p.model.thread for p in screws}.values()]
    ends = _axis_pass(mesh, regions, axes, threads.ON_TOL + mesh.noise) if axes else []
    thread_axes = axes

    # Pass 0b: surfaces on the axes found so far (a lug round its screw hole), before
    # pairs of facets can propose surfaces of their own: on a coarse, rounded-off mesh a
    # strip of three or four facets fits a cylinder of almost any radius and tilt.
    features += _axis_pass(mesh, regions, _distinct_axes(features, mesh), AXIS_NOISE * mesh.noise)

    # Fillets between flat faces (fillets.py): a cylinder touching both faces, only its
    # radius to find, with radii found already (holes, pins, rounded corners) as the
    # likeliest. Before pass 1, which would cut a coarsely meshed fillet into strips
    # that each fit some cylinder of their own.
    from . import fillets
    progress.step("finding rounded edges between flat faces")
    taken = np.zeros(len(mesh.farea), bool)
    for f in features + screws + ends:
        taken[f.facets] = True
    known = [r for r in (_radius(f.model) for f in features) if r]
    rolled = fillets.find(mesh, regions, taken, known, walls=features)
    claimed = [int(f) for x in rolled for f in x.facets]
    for region in regions:
        region.free &= ~np.isin(region.facets, claimed)
    rolled += _mirror_pass(mesh, regions, rolled)
    for x in rolled:
        x.__dict__["rolled"] = True     # (pinned down by the faces it joins: see _merge_same_surface)
    features += rolled

    # Pass 1: from each pair of neighbouring facets, work out what surface they're on.
    progress.step("finding curved surfaces")
    features += _pairs_pass(mesh, regions)
    features += _mirror_pass(mesh, regions, features)

    # Pass 2: tori (and anything missed) around the axes of the patches found so far.
    # A curved rounded edge always shares its axis with a neighbouring hole, pin or
    # rounded corner, so each new patch can unlock its neighbours.
    progress.step("finding rounded edges round holes and pins")
    done = []
    while True:
        # each round only tries the axes found since the last one
        axes = [x for x in _distinct_axes(features, mesh)
                if not any(abs(x[1] @ d) > math.cos(math.radians(0.5))
                           and np.linalg.norm(np.cross(x[0] - a, d)) < TOL * 4 for a, d, _ in done)]
        if not axes:
            break
        done += axes
        # (on a known axis only the radius is free, so a file that rounded its corners
        # can be allowed that much more: the CAD program's own export is often as loose)
        features += _axis_pass(mesh, regions, axes, AXIS_NOISE * mesh.noise)
    features += _lands(mesh, regions, _distinct_axes(features, mesh) + thread_axes)
    features += _mirror_pass(mesh, regions, features)
    if mesh.mirror_skip is not None:
        # (from here on everything left is searched, on both sides: whatever the mirror
        # didn't explain, an asymmetric detail or a surface grown differently)
        mesh.mirror_skip = None
        mesh.__dict__.pop("shared_as", None)
    progress.step("finding what is left")
    features += _loose_pass(mesh, regions, features)
    features = _band_tori(mesh, regions, features)
    # (the thread's pieces ahead of what was found round its ends: if the solid won't
    # check out with everything, the search for culprits keeps what comes first)
    return mesh, _fill_pinholes(mesh, _merge_same_surface(mesh, features)) + screws + ends, _mesh_tol


def _tube_fit(P, model, steps=30):
    """Least-squares torus, cylinder or cone through points P, starting from `model`.

    Gauss-Newton on the fewest parameters: the axis tilted by two small angles, the
    centre moved across it (and along it for a torus), and the radii (for a cone, the
    radius where the axis point is and how fast it grows).
    """
    torus = not model.line
    cone = not torus and model.line[1] != 0
    d0, e1, e2 = _frame(model.d)
    if torus:
        rc, zc, r = model.circle
        c0, x = model.a + zc * d0, np.array([0, 0, 0, 0, 0, rc, r], float)
    elif cone:
        c0, x = model.a, np.array([0, 0, 0, 0, model.line[0], model.line[1]], float)
    else:
        c0, x = model.a, np.array([0, 0, 0, 0, model.line[0]], float)

    def residuals(X):
        """Residuals for each row of parameters X (m, n): (m, len(P)). All the nudged
        parameter sets of a Jacobian go through at once."""
        d = d0 + X[:, 0:1] * e1 + X[:, 1:2] * e2
        d /= np.sqrt((d * d).sum(axis=1))[:, None]
        c = c0 + X[:, 2:3] * e1 + X[:, 3:4] * e2 + (X[:, 4:5] * d0 if torus else 0)
        v = P[None] - c[:, None]
        z = np.einsum("mpk,mk->mp", v, d)
        w = v - z[..., None] * d[:, None]
        rho = np.sqrt((w * w).sum(axis=2))
        if torus:
            return np.hypot(rho - X[:, 5:6], z) - X[:, 6:7]
        if cone:
            return (rho - X[:, 4:5] - X[:, 5:6] * z) / np.hypot(1, X[:, 5:6])
        return rho - X[:, 4:5]

    def res(x):
        return residuals(x[None])[0]

    f = res(x)
    for _ in range(steps):
        h = 1e-7 * np.maximum(1.0, np.abs(x))
        J = ((residuals(x + np.diag(h)) - f) / h[:, None]).T
        step = np.linalg.lstsq(J, -f, rcond=None)[0]
        x2 = x + step
        f2 = res(x2)
        while f2 @ f2 > f @ f and np.abs(step).max() > 1e-14:
            step /= 2
            x2 = x + step
            f2 = res(x2)
        if f2 @ f2 > f @ f:
            break
        x, f = x2, f2
        if np.abs(step).max() < 1e-12:
            break
    if not np.all(np.isfinite(x)):
        return None
    d = d0 + x[0] * e1 + x[1] * e2
    d /= np.linalg.norm(d)
    c = c0 + x[2] * e1 + x[3] * e2
    if torus:
        c = c + x[4] * d0
        return Revolved(c, d, circle=(abs(x[5]), float(c @ d), abs(x[6])))
    if cone:
        # the model measures heights from the axis point nearest the origin
        return Revolved(c, d, line=(float(x[4] - x[5] * (c @ d)), float(x[5])))
    return Revolved(c, d, line=(abs(x[4]), 0))


def _band_tori(mesh, regions, features):
    """Replace chains of small pieces by the torus (or cylinder) they approximate.

    A rounded edge that follows a curve, with no neighbouring hole or pin to supply its
    axis, is easily cut into many small pieces of the fillet radius: short cylinder
    strips (the idea of the stlToSolid project) or, on coarse meshes, little "spheres"
    (a short stretch of a torus fits a sphere of its tube radius too). The pieces' centres
    run along the torus's centre circle, so consecutive pieces are fitted as one torus,
    and the run is regrown as a single patch.
    """
    global _loose

    def piece(f):
        """(radius, point on the tube's centre line) for a small piece, else None."""
        m = f.model
        if isinstance(m, Sphere):
            P = mesh.pts[_uniq(np.concatenate([mesh.fverts[x] for x in f.facets]))]
            return (m.r, m.c) if np.ptp(P, axis=0).max() <= 2.5 * m.r else None
        if isinstance(m, Revolved) and m.line and m.line[1] == 0 and f.span < TWO_PI:
            if f.hi - f.lo <= 4 * m.line[0]:
                return m.line[0], m.a + (f.lo + f.hi) / 2 * m.d
        return None

    small = {k: piece(f) for k, f in enumerate(features)}
    small = {k: v for k, v in small.items() if v is not None}
    if len(small) < 3:
        return features
    keys = list(small)
    verts = {k: set(np.concatenate([mesh.fverts[x] for x in features[k].facets]).tolist()) for k in keys}
    link = {k: [] for k in keys}
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            ra, rb = small[a][0], small[b][0]
            if abs(ra - rb) <= max(0.05 * ra, 2 * _tol()) and verts[a] & verts[b]:
                link[a].append(b)
                link[b].append(a)
    region_of = {}
    for region in regions:
        for pos, f in enumerate(region.facets):
            region_of[int(f)] = (region, pos)

    def ordered(chain):
        """The chain's pieces in order along it (a walk from one end)."""
        members = set(chain)
        ends = [k for k in chain if len([j for j in link[k] if j in members]) <= 1]
        k = ends[0] if ends else chain[0]
        out, seen = [k], {k}
        while True:
            nxt = [j for j in link[k] if j in members and j not in seen]
            if not nxt:
                break
            # the nearest neighbour keeps the walk on the chain where it branches
            k = min(nxt, key=lambda j: np.linalg.norm(small[j][1] - small[out[-1]][1]))
            out.append(k)
            seen.add(k)
        return out

    def points(ks):
        return mesh.pts[_uniq(np.concatenate([mesh.fverts[x] for k in ks for x in features[k].facets]))]

    def tube(ks):
        """The torus or cylinder through these pieces, if their mesh corners lie on it."""
        C = np.array([small[k][1] for k in ks])
        r = float(np.mean([small[k][0] for k in ks]))
        mid = C.mean(axis=0)
        vt = _right_vectors(C - mid)
        guesses = [Revolved(mid, vt[0], line=(r, 0))]           # a straight fillet
        if len(ks) >= 3:
            _, e1, e2 = _frame(vt[2])
            c = _circle2d((C - mid) @ e1, (C - mid) @ e2)
            if c is not None and c[2] < 1e3 * r:
                centre = mid + c[0] * e1 + c[1] * e2
                guesses.append(Revolved(centre, vt[2], circle=(c[2], float(centre @ _frame(vt[2])[0]), r)))
        P = points(ks)
        fitted = [m for m in (_tube_fit(P, g) for g in guesses) if m is not None]
        fitted = [(np.abs(m.signed(P)).max(), k, m) for k, m in enumerate(fitted)]
        if not fitted:
            return None
        err, _, model = min(fitted)
        if err > 2 * _tol():
            return None
        if model.circle and model.circle[0] <= model.circle[2]:
            return None                                    # not a tube round a hole in the middle
        return model

    drop, added, seen = set(), [], set()
    for start in keys:
        if start in seen or len(link[start]) == 0:
            continue
        chain, stack = [], [start]
        seen.add(start)
        while stack:
            k = stack.pop()
            chain.append(k)
            for j in link[k]:
                if j not in seen:
                    seen.add(j)
                    stack.append(j)
        if len(chain) < 3:
            continue
        run_order = ordered(chain)
        i = 0
        while i + 3 <= len(run_order):
            # the longest run from i that one tube explains
            j, model = i + 3, tube(run_order[i:i + 3])
            if model is None:
                i += 1
                continue
            while j < len(run_order):
                better = tube(run_order[i:j + 1])
                if better is None:
                    break
                model, j = better, j + 1
            run = run_order[i:j]
            i = j
            facets = np.concatenate([features[k].facets for k in run])
            if int(facets[0]) not in region_of:
                continue
            region, seed = region_of[int(facets[0])]
            positions = [region_of[int(f)][1] for f in facets
                         if int(f) in region_of and region_of[int(f)][0] is region]
            region.free[positions] = True
            _loose = True
            try:
                got = _candidate(mesh, region, model, [seed], axis_fixed=True)
            finally:
                _loose = False
            area = float(mesh.farea[facets].sum())
            if got and got[0].model.kind == model.kind and got[2] >= 0.9 * area:
                feature, idx, _ = got
                region.free[idx] = False
                drop.update(run)
                # pieces the new patch swallowed whole go too
                taken = set(region.facets[idx].tolist())
                drop.update(k for k in keys if set(features[k].facets.tolist()) <= taken)
                added.append(feature)
            else:
                region.free[positions] = False
    if not added:
        return features
    return [f for k, f in enumerate(features) if k not in drop] + added


def _loose_pass(mesh, regions, features):
    """Pass 3: whatever curved facets are left, accept any surface that explains them."""
    return _pairs_pass(mesh, regions, _distinct_axes(features, mesh))


PARALLEL_FACETS = 3000  # unexplained facets needed before a pass is shared out among worker processes
SLICE_FACETS = 400      # a region bigger than this is shared out in slices of this many facets


def _pairs_pass(mesh, regions, axes=None):
    """Pass 1 (axes None) or the loose pass (axes: the axes found so far): from each pair
    of neighbouring facets, work out what surface they're on (in the loose pass,
    accepting any surface that explains them, including ones round the axes found)."""
    return _seed_pass(mesh, regions, ("pairs", axes, _mesh_tol))


def _axis_patches(mesh, regions, axes, lines_only=False):
    """Patches on the given axes round each free facet (with lines_only, cylinders and
    cones only), held to the tolerance in force."""
    return _seed_pass(mesh, regions, ("axis", axes, _mesh_tol, lines_only, _axis_vouched))


# A seed pass goes through each region's facets in order, trying the surfaces each
# suggests (with a neighbour, for "pairs"; round known axes, for "axis") and claiming
# the facets of every patch found. `how` says which, with what it needs: ("pairs",
# axes or None, tolerance) or ("axis", axes, tolerance, lines only, axis vouched).

def _seed_pass(mesh, regions, how):
    """The features of a seed pass over every region, in region order.

    On a big mesh this is shared out among worker processes (workers.py), with exactly
    the result of going through it in order. Regions don't affect one another, so each
    mid-sized one is simply done whole in a worker. A big one is cut into slices of
    facets, each gone through in order by a worker from the region as it stood when the
    pass began, noting which facets each try looked at; here the slices are then gone
    through in order, taking a worker's answer wherever those facets still stand as the
    worker saw them, and trying again here where a patch from elsewhere changed them.
    """
    work = [k for k, r in enumerate(regions) if r.free.sum() >= 2]
    results, pool, jobs = {}, None, {}
    if sum(int(regions[k].free.sum()) for k in work) >= PARALLEL_FACETS:
        from . import workers
        pool = workers.get()
    if how[0] == "axis" or how[1] is not None:         # (passes that start with nothing explored)
        for region in regions:
            region.explored[:] = False
    start = {k: (regions[k].free.copy(), regions[k].explored.copy()) for k in work}
    if pool is not None:
        try:
            if "shared_as" not in mesh.__dict__:
                mesh.shared_as = workers.share(mesh)
            order = sorted(work, key=lambda k: -len(regions[k].facets))
            # (each region's neighbour lists as they are here: rebuilt from the mesh's
            # sets in another process they can come out in another order, and the order
            # decides what is tried)
            big = [k for k in order if len(regions[k].facets) >= 50]
            state = workers.share({k: (regions[k].facets, regions[k].nbrs, regions[k].free, regions[k].explored,
                                       regions[k].__dict__.get("tube_fits", {}))
                                   for k in big})
            for k in big:
                n = len(regions[k].facets)
                if n <= SLICE_FACETS:
                    jobs[k] = pool.submit(_seed_job, (mesh.shared_as, state, how, k))
                else:
                    jobs[k] = [pool.submit(_seed_slice, (mesh.shared_as, state, how, k, lo, min(lo + SLICE_FACETS, n)))
                               for lo in range(0, n, SLICE_FACETS)]
            for k in order:             # (the small ones meanwhile here)
                if k not in jobs:
                    results[k] = _seed_region(mesh, regions[k], how)
            for k in big:
                if isinstance(jobs[k], list):
                    results[k] = _seed_replay(mesh, regions[k], how, jobs[k])
                else:
                    found, free, explored, fits = jobs[k].result()
                    regions[k].free[:], regions[k].explored[:] = free, explored
                    regions[k].tube_fits = fits
                    results[k] = found
        except Exception:
            workers.broken()            # (then done here, from where they started)
            for k in jobs:
                regions[k].free[:], regions[k].explored[:] = start[k]
                results.pop(k, None)
    for k in work:
        if k not in results:
            results[k] = _seed_region(mesh, regions[k], how)
    return [f for k in sorted(results) for f in results[k]]


class _Settings:
    """The module settings a seed pass runs under, put in place for its duration."""

    def __init__(self, how):
        self.how = how

    def __enter__(self):
        global _mesh_tol, _loose, _axis_vouched
        self.saved = _mesh_tol, _loose, _axis_vouched
        if self.how[0] == "pairs":
            _mesh_tol, _loose = self.how[2], self.how[1] is not None
        else:
            _mesh_tol, _axis_vouched = self.how[2], self.how[4]
        return self

    def __exit__(self, *exc):
        global _mesh_tol, _loose, _axis_vouched
        _mesh_tol, _loose, _axis_vouched = self.saved


def _seeds(region, i, how):
    """What facet i is tried with: each of its first three neighbours ("pairs"), or on
    its own ("axis"); nothing if it mirrors a facet on the side searched."""
    skip = region.mesh.__dict__.get("mirror_skip")
    if skip is not None and skip[region.facets[i]]:
        return ()
    return region.nbrs[i][:3] if how[0] == "pairs" else (None,)


def _mirror_pass(mesh, regions, features):
    """The surfaces found on the side searched, mirrored onto the other (symmetry.py):
    for each, the free facets there lying on its mirror image (as closely as it lies on
    its own facets, and facing the same way) and within its mirrored outline, made a
    patch of that surface. Taken by footprint, not grown: a fillet found loosely
    (fillets.py) wouldn't regrow at the mesh tolerance."""
    skip = mesh.__dict__.get("mirror_skip")
    if skip is None or not features:
        return []
    from . import symmetry
    from scipy.spatial import cKDTree
    where = {}
    for region in regions:
        for i, f in enumerate(region.facets.tolist()):
            where[f] = (region, i)
    free = np.zeros(len(mesh.farea), bool)
    for f, (region, i) in where.items():
        free[f] = region.free[i]
    for f in features:
        free[f.facets] = False      # (whatever a patch has, as the regions may not say)
    found = []
    for f in features:
        m = symmetry.mirror_model(f.model, mesh.mirror)
        if m is None:
            continue
        V = np.unique(np.concatenate([mesh.fverts[a] for a in f.facets]))
        Pm = symmetry.mirror_point(mesh.pts[V], mesh.mirror)
        lo, hi = Pm.min(axis=0) - _tol(), Pm.max(axis=0) + _tol()
        cand = np.nonzero(free & skip & np.all((mesh.fcent >= lo) & (mesh.fcent <= hi), axis=1))[0]
        if not len(cand):
            continue
        # (inside its mirrored outline: each facet's mirrored middle within the size of
        # the nearest of its own facets; and as close to it as its own corners lie: a
        # cone held to its patch's worst gap took the far half of a dome beside it)
        size = np.array([float(np.ptp(mesh.pts[mesh.fverts[a]], axis=0).max()) for a in f.facets])
        near_d, near_i = cKDTree(mesh.fcent[f.facets]).query(symmetry.mirror_point(mesh.fcent[cand], mesh.mirror))
        cand = cand[near_d <= size[near_i] + _tol()]
        tol = max(_tol(m), 1.1 * float(np.abs(f.model.signed(mesh.pts[V])).max()))
        facing = np.sign(np.einsum("ij,ij->i", mesh.fn[f.facets], f.model.normal(mesh.fcent[f.facets])) @ mesh.farea[f.facets])
        keep = []
        for a in cand.tolist():
            C = mesh.pts[mesh.fverts[a]]
            if np.abs(m.signed(C)).max() <= tol and np.sign(mesh.fn[a] @ m.normal(mesh.fcent[a][None])[0]) == facing:
                keep.append(a)
        if len(keep) < 3 or mesh.farea[keep].sum() < MIRROR_SHARE * mesh.farea[f.facets].sum():
            continue
        got = _described(mesh, m, np.array(keep), f.convex)
        if got is None:
            continue
        free[keep] = False
        for a in keep:
            region, i = where[a]
            region.free[i] = False
        found.append(got)
    return found

def _open(region, i, j, how):
    """Is the seed (i, j) to be tried, the region standing as it does?"""
    if region.explored[i] or not region.free[i]:
        return False
    if how[0] == "pairs":
        return bool(region.free[j])
    return sum(1 for k in region.nbrs[i] if region.free[k]) >= 1


def _seed_try(mesh, region, i, j, how):
    """The best patch grown from seed (i, j), or None."""
    if region.reads is not None:        # (see _seed_slice)
        region.reads.append(np.array([i] + region.nbrs[i] + ([j] + region.nbrs[j] if j is not None else [])))
    if how[0] == "pairs":
        return _pair_seed(mesh, region, i, j, how[1])
    axes, lines_only = how[1], how[3]
    near = [i] + [k for k in region.nbrs[i] if region.free[k]]
    pts = region.points(near)
    found = []
    for axis in _near(axes, pts):
        model = _on_axis(axis, pts)
        if lines_only and model is not None and not model.line:
            continue
        if _fits(model, pts):
            found.append(_candidate(mesh, region, model, [i], axis_fixed=True))
    if _axis_vouched:
        # (the wider allowance lets a cone a degree off a cylinder fit a strip of
        # it; a real cone, a chamfer or countersink, tapers far more)
        found = [c for c in found if not c or not c[0].model.line
                 or c[0].model.line[1] == 0 or abs(c[0].model.line[1]) >= math.tan(math.radians(MIN_VOUCHED_CONE_DEG))]
    return _best(found)


def _seed_region(mesh, region, how):
    """A seed pass over one region, in order: its features."""
    found_all = []
    with _Settings(how):
        if how[0] == "axis" or how[1] is not None:
            region.explored[:] = False
        for i in range(len(region.facets)):
            for j in _seeds(region, i, how):
                if not _open(region, i, j, how):
                    continue
                best = _seed_try(mesh, region, i, j, how)
                if best:
                    feature, idx, _ = best
                    region.free[idx] = False
                    found_all.append(feature)
                    break
    return found_all


def _worker_region(mesh_key, state_key, k):
    """In a worker: region k of the shared mesh as it stood when the pass began."""
    from . import workers
    mesh = workers.load(mesh_key)
    facets, nbrs, free, explored, fits = workers.load(state_key)[k]
    if workers.memo.get("state") != state_key:
        workers.memo.clear()
        workers.memo["state"] = state_key
    region = workers.memo.get(k)
    if region is None:
        region = workers.memo[k] = Region(mesh, facets)
        region.nbrs = nbrs
        region.tube_fits = dict(fits)
    region.free[:], region.explored[:] = free, explored
    return mesh, region


def _seed_job(job):
    """In a worker: _seed_region over one whole region."""
    mesh_key, state_key, how, k = job
    mesh, region = _worker_region(mesh_key, state_key, k)
    found = _seed_region(mesh, region, how)
    return found, region.free, region.explored, region.tube_fits


def _seed_slice(job):
    """In a worker: the seeds of a slice of a region's facets gone through in order, as
    _seed_region would, but from the region as it stood when the pass began (nothing
    found in the slices before this one): [(i, j, what was found, facets whose
    availability the try looked at, facets it marked explored)] for every seed tried."""
    mesh_key, state_key, how, k, lo, hi = job
    mesh, region = _worker_region(mesh_key, state_key, k)
    out = []
    with _Settings(how):
        try:
            for i in range(lo, hi):
                for j in _seeds(region, i, how):
                    if not _open(region, i, j, how):
                        continue
                    region.reads, region.marks = [], []
                    best = _seed_try(mesh, region, i, j, how)
                    out.append((i, j, best, _uniq(np.concatenate(region.reads)).astype(np.int32), region.marks))
                    if best:
                        region.free[best[1]] = False
                        break
        finally:
            region.reads = region.marks = None
    return out


def _seed_replay(mesh, region, how, slices):
    """Go through a region's seeds in order, taking the workers' answers (_seed_slice)
    wherever every facet a try looked at stands as it did for the worker: the same
    result as _seed_region. Only tries that something outside their own slice changed
    the ground for are made again here."""
    found_all = []
    start = region.free.copy()
    with _Settings(how):
        if how[0] == "axis" or how[1] is not None:
            region.explored[:] = False
        for c, job in enumerate(slices):
            tried = job.result()
            seen = start.copy()             # availability as the worker saw it
            at = 0
            for i in range(c * SLICE_FACETS, min((c + 1) * SLICE_FACETS, len(region.facets))):
                done = False
                for j in _seeds(region, i, how):
                    got = None
                    if at < len(tried) and tried[at][0] == i and tried[at][1] == j:
                        got = tried[at]
                        at += 1
                    if not done and _open(region, i, j, how):
                        if got is not None and (seen[got[3]] == region.free[got[3]]).all():
                            best = got[2]
                            for idx in got[4]:
                                region.explored[idx] = True
                        else:
                            best = _seed_try(mesh, region, i, j, how)
                        if best:
                            feature, idx, _ = best
                            region.free[idx] = False
                            found_all.append(feature)
                            done = True
                    if got is not None and got[2]:
                        seen[got[2][1]] = False     # (the worker carried on from its own find)
    return found_all


def _pair_seed(mesh, region, i, j, axes=None):
    """The best patch grown from facets i and j (pass 1), or in the loose pass (axes:
    the axes found so far) from any surface that explains them, those axes included."""
    if axes is None:
        return _seed_candidates(mesh, region, i, j)
    found = [_seed_candidates(mesh, region, i, j)]
    near = [i, j] + [k for k in region.nbrs[i] + region.nbrs[j] if region.free[k]]
    pts = region.points(near)
    for axis in _near(axes, pts):
        m = _on_axis(axis, pts)
        if _fits(m, pts):
            found.append(_candidate(mesh, region, m, [i, j], axis_fixed=True))
    return _best(found)


def _simple_outline(mesh, fids):
    """True if no vertex appears more than once on the patch's outline loops."""
    T = np.concatenate([mesh.tris[mesh.ftris[f]] for f in fids])
    uniq, counts = _edge_counts(T, len(mesh.pts))
    rim = uniq[counts == 1].ravel()
    return len(rim) == 0 or np.bincount(rim).max() <= 2


# ---------------------------------------------------------------- snapping

def _circle_fixed_r(x, y, r, cx, cy):
    """Centre of the radius-r circle best fitting points (x, y), starting from (cx, cy)."""
    for _ in range(30):
        dx, dy = x - cx, y - cy
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        step = np.linalg.lstsq(np.c_[-dx / d, -dy / d], -(d - r), rcond=None)[0]
        cx, cy = cx + step[0], cy + step[1]
        if np.abs(step).max() < 1e-13:
            break
    return cx, cy


def _with_radius(model, P, r):
    """The same kind of surface with radius r, repositioned to fit points P best."""
    if isinstance(model, Sphere):
        c = model.c
        for _ in range(30):
            v = P - c
            d = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
            step = np.linalg.lstsq(-v / d[:, None], -(d - r), rcond=None)[0]
            c = c + step
            if np.abs(step).max() < 1e-13:
                break
        return Sphere(c, r)
    if model.line and model.line[1] == 0:           # cylinder: keep the direction
        x, y = P @ model.e1, P @ model.e2
        cx, cy = _circle_fixed_r(x, y, r, model.a @ model.e1, model.a @ model.e2)
        return Revolved(cx * model.e1 + cy * model.e2, model.d, line=(r, 0))
    if model.circle:                                # torus or cap: keep the axis
        rho, z, _ = model.local(P)
        rc, zc, _ = model.circle
        if rc == 0:                                 # sphere cap: slide the centre along the axis
            for _ in range(30):
                d = np.maximum(np.hypot(rho, z - zc), 1e-12)
                J = (-(z - zc) / d)[:, None]
                step = float(np.linalg.lstsq(J, -(d - r), rcond=None)[0][0])
                zc += step
                if abs(step) < 1e-13:
                    break
            return Revolved(model.a, model.d, circle=(0.0, zc, r))
        rc, zc = _circle_fixed_r(rho, z, r, rc, zc)
        return Revolved(model.a, model.d, circle=(rc, zc, r))
    return None


def _radius(model):
    if isinstance(model, Sphere):
        return model.r
    if not isinstance(model, Revolved):
        return None
    if model.line:
        return model.line[0] if model.line[1] == 0 else None
    return model.circle[2]


def _rebuilt(mesh, feature, model):
    """The feature on a new surface, if that surface still explains its facets."""
    global _loose
    for loose in (False, True):
        _loose = loose
        try:
            f = _feature(mesh, model, feature.facets, feature.convex)
        finally:
            _loose = False
        if f is not None:
            return f
    return None


def half_rings(mesh, feature):
    """A patch that runs all the way round its axis but is cut to shape (its outline
    wraps the surface's seam) as two half rings on the same surface, or None."""
    model = feature.model
    if feature.kind != "trimmed" or not isinstance(model, Revolved) or feature.span < TWO_PI:
        return None
    centres = np.array([mesh.pts[np.unique(mesh.fverts[f])].mean(axis=0) for f in feature.facets])
    u = model.angle(model.local(centres)[2])
    global _loose, _anchored
    out = []
    for side in (np.cos(u) >= 0, np.cos(u) < 0):
        _loose, _anchored = True, True     # the whole ring has already passed the checks
        try:
            half = _feature(mesh, model, feature.facets[side], feature.convex) if side.any() else None
        finally:
            _loose, _anchored = False, False
        if half is None:
            return None
        out.append(half)
    return out


def snap(mesh, features, round_unit=None):
    """Design intent: equal radii made exactly equal, near-axis-aligned axes made exact,
    and (with round_unit "mm" or "inch") radii set to round values. Each change is kept
    only if the patch's mesh corners still lie on the surface; otherwise it is undone."""
    from .sizing import roundness, INCH
    tol = 2 * _tol()
    points = [mesh.pts[_uniq(np.concatenate([mesh.fverts[f] for f in x.facets]))] for x in features]
    changed = 0

    def fits(model, P):
        return model is not None and np.abs(model.signed(P)).max() <= tol

    # axes within a hair of X, Y or Z become exactly X, Y or Z
    for k, f in enumerate(features):
        m = f.model
        if not isinstance(m, Revolved):
            continue
        axis = np.eye(3)[np.argmax(np.abs(m.d))] * np.sign(m.d[np.argmax(np.abs(m.d))])
        if abs(m.d @ axis) >= 1 - 1e-12 or abs(m.d @ axis) < math.cos(math.radians(0.1)):
            continue
        if m.line and m.line[1] == 0:
            new = _cylinder(axis, points[k])
            new = _with_radius(new, points[k], m.line[0]) if new else None
        else:
            new = Revolved(m.a, axis, line=m.line, circle=m.circle)
        if fits(new, points[k]):
            g = _rebuilt(mesh, f, new)
            if g:
                features[k], changed = g, changed + 1

    # radii: group equal ones, then optionally round the group's value
    radii = [(k, _radius(f.model)) for k, f in enumerate(features)]
    radii = sorted([(r, k) for k, r in radii if r], key=lambda x: x[0])
    groups, current = [], []
    for r, k in radii:
        if current and r - current[-1][0] > 2e-3 * r:
            groups.append(current)
            current = []
        current.append((r, k))
    if current:
        groups.append(current)
    for group in groups:
        weights = np.array([mesh.farea[features[k].facets].sum() for _, k in group])
        target = float(np.average([r for r, _ in group], weights=weights))
        if round_unit:
            scale = INCH if round_unit == "inch" else 1.0
            for step in ((1 / 16, 0.05, 1 / 64, 0.01) if round_unit == "inch" else (0.5, 0.1)):
                nice = round(target / scale / step) * step * scale
                if nice > 0 and roundness(nice, round_unit) and abs(nice - target) <= 2e-3 * target:
                    target = nice
                    break
        for r, k in group:
            if r == target:
                continue
            new = _with_radius(features[k].model, points[k], target)
            if fits(new, points[k]):
                g = _rebuilt(mesh, features[k], new)
                if g:
                    features[k], changed = g, changed + 1
    # (now the surfaces are final, leftover facets lying on them can join them)
    _absorb_straddlers(mesh, features)
    return changed


def _strip_of(model, other):
    """Is `other` nearly the surface `model` is: both cylinders (or both tori) round axes
    within SAME_TILT_DEG of each other, their radii within SAME_SHARE; or both spheres,
    radii within SAME_SHARE and centres that near?"""
    if isinstance(model, Sphere) and isinstance(other, Sphere):
        return (abs(model.r - other.r) <= SAME_SHARE * model.r
                and np.linalg.norm(model.c - other.c) <= SAME_SHARE * model.r)
    if not (isinstance(model, Revolved) and isinstance(other, Revolved)):
        return False
    if abs(model.d @ other.d) < math.cos(math.radians(SAME_TILT_DEG)):
        return False
    if model.line and other.line and model.line[1] == 0 and other.line[1] == 0:
        return abs(model.line[0] - other.line[0]) <= SAME_SHARE * model.line[0]
    if model.circle and other.circle:
        return (abs(model.circle[2] - other.circle[2]) <= SAME_SHARE * model.circle[2]
                and abs(model.circle[0] - other.circle[0]) <= SAME_SHARE * max(model.circle[0], model.circle[2]))
    return False


def _holds(mesh, model, facets, tol, convex):
    """Do these facets' corners all lie within tol of the model's surface, each facet
    facing along it (within NORMAL_DEG) the way a patch of it that is convex (or not) does?"""
    P = mesh.pts[_uniq(np.concatenate([mesh.fverts[f] for f in facets]))]
    if np.abs(model.signed(P)).max() > tol:
        return False
    facing = np.einsum("ij,ij->i", mesh.fn[facets], model.normal(mesh.fcent[facets]))
    return bool(((facing if convex else -facing) >= math.cos(math.radians(NORMAL_DEG))).all())


def _described(mesh, model, fids, convex):
    """_feature, else the same with its outline rules waived: for a patch whose surface
    is pinned down already (taking in its neighbours: see _merge_same_surface)."""
    global _loose, _anchored
    got = _feature(mesh, model, fids, convex)
    if got is None:
        _loose, _anchored = True, True
        try:
            got = _feature(mesh, model, fids, convex)
        finally:
            _loose, _anchored = False, False
    return got


def carve_exact(mesh, feature, points, reach):
    """A big cylinder, cone, torus or sphere patch blamed for trouble at these points less
    its facets within reach of them (left flat): [the patch], or None if it is small, or
    would lose half of itself. A bore cut across by another failed only where the slot's
    edge, both bores and the fillets between them met in a few small blends, and dropped
    whole its 232 facets went back to triangles."""
    if (points is None or not len(points) or len(feature.facets) < EXACT_CARVE_MIN
            or not isinstance(feature.model, (Revolved, Sphere))):
        return None
    from scipy.spatial import cKDTree
    facets = feature.facets
    size = np.array([np.linalg.norm(mesh.pts[mesh.fverts[a]] - mesh.fcent[a], axis=1).max() for a in facets])
    d, _ = cKDTree(np.asarray(points, float)).query(mesh.fcent[facets])
    keep = facets[d > reach + size]
    if len(keep) == len(facets) or len(keep) < len(facets) / 2:
        return None
    got = _described(mesh, feature.model, keep, feature.convex)
    return None if got is None else [got]


def _fill_pinholes(mesh, features):
    """Hand each small group of facets no patch has (PINHOLE_FACETS at most) that only one
    patch borders smoothly, every corner of it a corner of that patch and its middle as
    near the surface as the patch's own facets, to the patch. Triangles left out of a
    coarse fillet where it ends at a corner ball (a fan round one corner, alternately the
    fillet's and nobody's) pinched its outline into loops touching at that corner, and
    such a face wouldn't cut, nor would a blend over it fill (a comb's prong tips: four
    fillets of 97 facets went back to triangles, 1,064 -> 839 faces)."""
    owner = np.full(len(mesh.farea), -1)
    for k, f in enumerate(features):
        owner[f.facets] = k
    corners = [None] * len(features)
    grown = {}
    seen = set()
    for g0 in np.nonzero(owner < 0)[0].tolist():
        if g0 in seen:
            continue
        group, stack, around = [], [g0], set()
        seen.add(g0)
        while stack and len(group) <= PINHOLE_FACETS:
            g = stack.pop()
            group.append(g)
            for h in mesh.nbrs[g]:
                h = int(h)
                if owner[h] >= 0:
                    around.add(int(owner[h]))
                elif h not in seen:
                    seen.add(h)
                    stack.append(h)
        if stack or len(group) > PINHOLE_FACETS or len(around) != 1:
            continue
        k = around.pop()
        if corners[k] is None:
            corners[k] = set(np.concatenate([mesh.fverts[a] for a in features[k].facets]).tolist())
        f = features[k]
        if (all(set(mesh.fverts[g].tolist()) <= corners[k] for g in group)
                and np.abs(f.model.signed(mesh.fcent[group])).max() <= max(f.worst, _tol(f.model))):
            grown.setdefault(k, []).extend(group)
    for k, extra in grown.items():
        f = features[k]
        wider = _described(mesh, f.model, np.r_[f.facets, extra], f.convex)
        if wider is not None:
            features[k] = wider
    return features


def _sure(f):
    """Is the patch sure of its surface: found over SAME_FACETS facets, or a fillet rolled
    between two flat faces (fillets.py: the faces pin it down, however few its strips; a
    box's r 4 bottom fillet, 23 strips running its length, had the slivers along the
    bottom face fitted on their own as r 4.067)?"""
    return len(f.facets) >= SAME_FACETS or bool(f.__dict__.get("rolled"))


def _merge_same_surface(mesh, features):
    """Join neighbouring patches that lie on the same surface (a hole split in two by a
    seam in the mesh, say), so each becomes one face. Pairs are tried in order (each
    patch with those after it that share a corner, the first merge taking the first
    patch's place); pairs that touch are found through their corners, and a pair found
    not to merge is skipped from then on (its outcome depends only on the two features,
    the same objects until one merges): trying every pair again after every merge took
    the toolbox 214 s, now 7. After a merge the order goes on as if started over, which
    it once was: every pair before the merged patch has been refused already, so only
    the patches before it that touch it are tried with it (starting over and skipping
    the refused pairs took a 23,000-patch mesh 2 minutes of looking up corners)."""
    verts = [set(np.concatenate([mesh.fverts[f] for f in x.facets]).tolist()) for x in features]
    failed = set()      # (pairs refused a joint surface)
    refused = {}        # (id(a), id(b)) -> (a, b): pairs that didn't merge (kept alive, so ids stay theirs)
    users = {}          # corner -> ids of the features using it
    for x, vs in zip(features, verts):
        for v in vs:
            users.setdefault(v, set()).add(id(x))
    index = {id(x): k for k, x in enumerate(features)}     # (merged-away patches leave a None)
    features = list(features)

    def near(i, before):
        """Positions of the patches sharing a corner with patch i, after it or before it."""
        ids = set()
        for v in verts[i]:
            ids |= users[v]
        if before:
            return sorted(index[q] for q in ids if index.get(q, i) < i)
        return sorted(index[q] for q in ids if index.get(q, -1) > i)

    def merge(i, j):
        """Try patches i < j as one; if they merge, the joint patch takes i's place."""
        a, b = features[i], features[j]
        if (id(a), id(b)) in refused:
            return False
        # (one surface holding both is enough. And a cylinder found over
        # SAME_FACETS facets or more, sure of itself, takes in a strip beside it
        # fitted as nearly the same cylinder if its corners lie within SAME_TOL:
        # a bore cut across by another had the corners along the crossing up to
        # 12 um off, and held to the mesh tolerance came out as 44 strips, each a
        # cylinder 0.02-0.3 mm off over its narrow arc. Only such strips: taking
        # whatever lay that near cost the gutter mount 47 faces)
        holder = None
        for x, y in sorted(((a, b), (b, a)), key=lambda xy: -len(xy[0].facets)):
            if _sure(x) and _strip_of(x.model, y.model):
                ok = _holds(mesh, x.model, y.facets, max(SAME_TOL, _tol()), x.convex)
            else:
                ok = x.convex == y.convex and _holds(mesh, x.model, y.facets, _tol(), x.convex)
            if ok:
                holder = x
                break
        both = None
        if holder is not None:
            fids = np.concatenate([a.facets, b.facets])
            # (outline touching itself, e.g. equal holes crossing: no merge.) The
            # outline rules are waived for a sure holder, and for two pieces each
            # lying on the other's surface: each passed them alone. A box's 63 mm
            # r 2 bottom fillet came as 26 and 12 facets of one cylinder, their
            # long triangles interleaved, and together failed the outline rules;
            # the 12, touching one another only at corners, then couldn't be cut
            if _simple_outline(mesh, fids):
                other = b if holder is a else a
                sure = _sure(holder) or (
                    other.convex == holder.convex and _holds(mesh, other.model, holder.facets, _tol(), other.convex))
                both = (_described if sure else _feature)(mesh, holder.model, fids, holder.convex)
            if both is not None:
                new_verts = verts[i] | verts[j]
        if both is None:
            # (neither holds the other, or one did but the pair failed the
            # outline rules on its surface: one surface fitted to both, if exact,
            # pins it down well enough to waive them)
            pair = (a.facets.tobytes(), b.facets.tobytes())
            if pair in failed:
                refused[id(a), id(b)] = (a, b)
                return False
            joint = _joint_surface(mesh, a, b)
            both = (_described(mesh, joint[0], joint[1], a.convex)
                    if joint is not None and _simple_outline(mesh, joint[1]) else None)
            if both is None:
                failed.add(pair)
                refused[id(a), id(b)] = (a, b)
                return False
            new_verts = set(np.concatenate([mesh.fverts[f] for f in both.facets]).tolist())
        # (both takes a's place, b goes)
        for v in verts[i]:
            users[v].discard(id(a))
        for v in verts[j]:
            users[v].discard(id(b))
        for v in new_verts:
            users.setdefault(v, set()).add(id(both))
        del index[id(a)], index[id(b)]
        index[id(both)] = i
        verts[i], verts[j] = new_verts, None
        features[i], features[j] = both, None
        return True

    i = 0
    while i < len(features):
        if features[i] is None:
            i += 1
            continue
        for j in near(i, False):
            if merge(i, j):
                # (started over, the order would come to this patch again only after
                # trying each patch before it that touches it with it, in order)
                k = 0
                while k is not None:
                    k = next((k for k in near(i, True) if merge(k, i)), None)
                    if k is not None:
                        i = k
                break
        else:
            i += 1
    return [x for x in features if x is not None]


def _joint_surface(mesh, a, b):
    """One cylinder or torus through every corner of two neighbouring pieces fitted
    as nearly the same surface (_strip_of), each corner within the mesh tolerance and each
    facet facing along it; else None. Pieces grown from narrow strips get their surface
    from those strips (a refit needs whole rings of corners round the axis), a few
    hundredths off over a short arc, and so don't lie on each other's: a 23 mm ramp
    whose sides flare (no two strip ends at one height) came out as five pieces, r 22.99
    to 23.07, where every corner lay within 0.0003 mm of the one true cylinder."""
    # (cylinders and tori only: over the small patches a sphere is found on, any radius
    # within a percent fits as well, and two such pieces of the solar bracket, r 9.09
    # and 9.19, joined as r 9.14 failed to build and took their neighbours with them)
    if (a.convex != b.convex or not isinstance(a.model, Revolved) or not isinstance(b.model, Revolved)
            or not _strip_of(a.model, b.model)):
        return None
    fids = np.concatenate([a.facets, b.facets])
    big = a if len(a.facets) >= len(b.facets) else b
    # (a piece fitted a little off can have taken in a facet or two of the round beside
    # it, which the true surface leaves out: those are left out and fitted afresh)
    stray = max(1, int(JOINT_STRAY * min(len(a.facets), len(b.facets))))

    def fit(facets):
        P = mesh.pts[_uniq(np.concatenate([mesh.fverts[f] for f in facets]))]
        model = _tube_fit(P, big.model)
        if model is None:
            return None, None
        off = np.array([np.abs(model.signed(mesh.pts[mesh.fverts[f]])).max() > _tol() for f in facets])
        return model, off

    model, off = fit(fids)
    if model is None or off.sum() > stray or not _strip_of(big.model, model):
        return None     # (over a short arc a far other radius fits too: 2.45 and 2.35 gave 1.52)
    keep = fids
    if off.any():
        keep = fids[~off]
        model, off = fit(keep)
        if model is None or off.any() or not _strip_of(big.model, model):
            return None
    if not _holds(mesh, model, keep, _tol(), a.convex):
        return None
    # (and it must reach at least a radius along or round its surface, or the corners
    # don't pin the radius down: specks of the latch's pocket corners, 0.13 mm of an
    # r 0.5 round, r 0.57 and 0.50, joined as r 0.49, and their faces left gaps)
    P = mesh.pts[_uniq(np.concatenate([mesh.fverts[f] for f in keep]))]
    rho, z, w = model.local(P)
    u = np.unwrap(np.sort(model.angle(w)))
    if model.line:
        reach, r = max(np.ptp(z), rho.mean() * min(np.ptp(u), TWO_PI)), model.line[0]
    else:
        rc, zc, r = model.circle
        tube = np.arctan2(z - zc, rho - rc)
        reach = max(rc * min(np.ptp(u), TWO_PI), r * np.ptp(np.unwrap(np.sort(tube))))
    if reach < r:
        return None
    return model, keep


def _absorb_straddlers(mesh, features):
    """Hand each leftover facet that lies on a neighbouring patch's surface to it.

    Where a finely cut curve runs straight on into another (a bent pipe's crest where
    the straight run meets the bend) a long strip of one and a sliver of the other can
    be flat together to well under a micron, and are joined into one facet that neither
    patch can take. Left flat, such a facet is a hairline face between two curved ones
    that the sewing can't close; so is a needle-thin triangle whose normal kept it out
    of the patch. Their corners do lie on the surface beside them, within a
    few tolerances; the patch is described afresh with it and keeps it if that holds (the
    surface is already pinned down by the patch, so its outline rules are waived as for
    fillets). Runs once the surfaces are final (after snap)."""
    global _loose, _anchored
    owner = np.full(len(mesh.farea), -1)
    for k, f in enumerate(features):
        owner[f.facets] = k

    # (again while any join, but then only facets every corner of which is the patch's
    # already: a needle can have only leftovers beside it until they join their patch.
    # A 53 mm needle 0.006 mm wide between a wall and its r 1.5 fillet, whose two strips
    # beside it joined the fillet only after it was passed over, was left a face of its
    # own, and the fillet blamed for the gap at it. Needles at a fillet's end, where
    # another crosses it, joined on later passes too, ran it past its end and left a gap)
    # (a patch's corners as a set, kept for as long as the patch is the same: worked out
    # afresh for every leftover beside it, they made this quadratic on a fine mesh)
    corners = {}

    def corners_of(k):
        got = corners.get(k)
        if got is None or got[0] is not features[k]:
            got = corners[k] = (features[k], set(np.concatenate([mesh.fverts[a] for a in features[k].facets]).tolist()))
        return got[1]

    joined, again = True, None
    while joined:
        joined, again = False, again is not None
        for g in np.nonzero(owner < 0)[0]:
            V = mesh.pts[mesh.fverts[g]]
            # (a needle-thin facet's own normal means nothing, as in Region)
            needle = mesh.farea[g] < 0.05 * np.ptp(V, axis=0).max() ** 2
            if len(mesh.ftris[g]) < 2 and not needle:
                continue    # an ordinary curved facet the passes left: smooth blends see to those
            C = mesh.pts[mesh.tris[mesh.ftris[g]]]
            mids = np.r_[((C + C[:, [1, 2, 0]]) / 2).reshape(-1, 3), C.mean(axis=1)]
            for k in sorted({owner[h] for h in mesh.nbrs[g]} - {-1}):
                f = features[k]
                if not isinstance(f.model, (Revolved, Sphere)):
                    continue
                if np.abs(f.model.signed(V)).max() > STRADDLE_STEPS * _tol(f.model):
                    continue
                if again and not set(mesh.fverts[g].tolist()) <= corners_of(k):
                    continue
                if not needle and abs(mesh.fn[g] @ f.model.normal(mesh.fcent[g][None])[0]) < math.cos(math.radians(NORMAL_DEG)):
                    continue
                # it must bow off the surface no more than the patch's own facets do (a real
                # flat, a D-flat on a shaft, has its corners on the surface too, but cuts deeper)
                if np.abs(f.model.signed(mids)).max() > max(1.5 * f.worst, STRADDLE_STEPS * _tol(f.model)):
                    continue
                wider = _feature(mesh, f.model, np.r_[f.facets, g], f.convex)
                if wider is None:
                    _loose, _anchored = True, True
                    try:
                        wider = _feature(mesh, f.model, np.r_[f.facets, g], f.convex)
                    finally:
                        _loose, _anchored = False, False
                if wider is not None:
                    features[k] = wider
                    owner[g] = k
                    joined = True
                    break
    return features


def _lands(mesh, regions, axes):
    """Narrow strips all on one cylinder round a known axis: the lands of a knurled
    or splined round between its grooves, exported a few flat facets each. Alone each
    is too narrow to trust (MIN_SPAN_DEG: a flat strip fits almost any big cylinder);
    together, every corner on one radius round an axis found already, they are what
    the round was drawn as (the gutter mount's knurled screw head: 30 lands of 3
    facets, 5 deg each, corners within 0.0004 mm of r 20, left to smooth blends)."""
    global _loose, _anchored
    tol = LAND_TOL + mesh.noise
    out = []
    for a, d, reach in axes:
        base = Revolved(a, d, line=(1.0, 0))
        strips = []
        for region in regions:
            idx = np.nonzero(region.free)[0]
            if not 2 <= len(idx) <= LAND_MAX_FACETS:
                continue
            rho, _, u = base.local(region.points(idx))
            if np.ptp(rho) > tol or rho.mean() > reach:
                continue
            # (each facet facing straight out from the axis, or straight in)
            c = mesh.fcent[region.facets[idx]] - a
            radial = c - np.outer(c @ d, d)
            radial /= np.linalg.norm(radial, axis=1)[:, None]
            facing = np.einsum("ij,ij->i", radial, region.n[idx])
            if np.abs(facing).min() < math.cos(math.radians(LAND_TILT_DEG)) or np.ptp(np.sign(facing)) > 0:
                continue
            u = np.unwrap(np.sort(np.mod(u, TWO_PI)))
            strips.append((region, idx, float(rho.mean()), bool(facing[0] > 0), float(np.ptp(u))))
        strips.sort(key=lambda x: x[2])
        while strips:
            group = [x for x in strips if abs(x[2] - strips[0][2]) <= tol and x[3] == strips[0][3]]
            strips = [x for x in strips if all(x is not g for g in group)]
            if len(group) < LAND_MIN or sum(x[4] for x in group) < math.radians(MIN_SPAN_DEG):
                continue
            model = Revolved(a, d, line=(float(np.mean([x[2] for x in group])), 0))
            for region, idx, _, convex, _ in group:
                _loose, _anchored = True, True
                try:
                    feature = _feature(mesh, model, region.facets[idx], convex)
                finally:
                    _loose, _anchored = False, False
                if feature is not None:
                    feature.vouched = True      # (by the others: not a stand-in, blends._scrap)
                    region.free[idx] = False
                    out.append(feature)
    return out


def corner_surface(mesh, facets):
    """A cylinder or cone feature through every corner of these facets, its axis fitted
    to the corners alone, or None. On a coarse export a gentle curve can be a few big
    triangles cutting across it at a slant: their normals aren't square to the axis, so
    the guesses from normals all go astray, and the area came out as fragments of
    cylinders with mismatched radii, then a smooth blend (the gutter mount's slope:
    14 facets turning 21 deg, every corner within 0.0036 mm of r 21.04)."""
    import scipy.optimize as so
    global _loose, _anchored
    fac = np.asarray(facets)
    if len(fac) < CORNER_FIT_FACETS:
        return None
    N = mesh.fn[fac]
    if math.degrees(math.acos(max(-1.0, min(1.0, float(_least_pair_dot(N, math.cos(math.radians(MIN_SPAN_DEG)))))))) < MIN_SPAN_DEG:
        return None
    P = mesh.pts[_uniq(np.concatenate([mesh.fverts[f] for f in fac]))]
    c0 = P.mean(axis=0)
    out_n = N.mean(axis=0)
    out_n /= max(np.linalg.norm(out_n), 1e-12)
    guesses = [np.linalg.eigh(N.T @ N)[1][:, 0], _right_vectors(P - c0)[0]]
    best = None
    for g in guesses:
        d, e1, e2 = _frame(g)

        def cylinder(x, d=d, e1=e1, e2=e2):
            dd = d + x[0] * e1 + x[1] * e2
            dd = dd / np.linalg.norm(dd)
            w = P - (c0 + x[2] * e1 + x[3] * e2)
            w -= np.outer(w @ dd, dd)
            return np.linalg.norm(w, axis=1) - x[4]
        for r0 in (5.0, 20.0, 80.0):
            for side in (1, -1):
                off = side * r0 * out_n
                fit = so.least_squares(cylinder, np.r_[0, 0, off @ e1, off @ e2, r0])
                if best is None or np.abs(fit.fun).max() < np.abs(best[0].fun).max():
                    best = (fit, d, e1, e2)
    fit, d, e1, e2 = best
    tol = CORNER_FIT_DEV + mesh.noise
    if np.abs(fit.fun).max() > tol or not 0 < fit.x[4] < CORNER_FIT_MAX_R:
        return None
    axis = d + fit.x[0] * e1 + fit.x[1] * e2
    axis /= np.linalg.norm(axis)
    model = Revolved(c0 + fit.x[2] * e1 + fit.x[3] * e2, axis, line=(float(fit.x[4]), 0))
    if (np.abs(model.signed(mesh.fcent[fac])) > MAX_SAG).any():
        return None
    region = Region(mesh, fac)
    ok, outward, _ = region._test(model, np.arange(len(fac)))
    if (outward != outward[0]).any():
        return None
    _loose, _anchored = True, True      # (the fit is the evidence: every corner on it)
    try:
        return _feature(mesh, model, fac, bool(outward[0]))
    finally:
        _loose, _anchored = False, False


def _distinct_axes(features, mesh=None):
    """[(point, direction, reach)]: each distinct axis, and how far from it a neighbouring
    rounded edge could plausibly lie (a few times the largest radius on that axis).

    A dome or ball contributes an axis through its centre square to each flat face it
    borders: a rounded edge between the two is a torus on that axis."""
    found = []
    for f in features:
        m = f.model
        if isinstance(m, Revolved):
            size = m.line[0] + abs(m.line[1]) * 10 if m.line else m.circle[0] + m.circle[2]
            found.append((m.a, m.d, size))
        elif isinstance(m, Sphere) and mesh is not None:
            inside = set(f.facets.tolist())
            P = mesh.pts[_uniq(np.concatenate([mesh.fverts[x] for x in f.facets]))]
            if np.ptp(P, axis=0).max() < m.r:
                continue                      # a small piece of something else
            border = {g for x in f.facets for g in mesh.nbrs[x] if g not in inside}
            big = [g for g in border if mesh.farea[g] > 0.05 * m.r * m.r]
            for g in big:
                found.append((m.c, mesh.fn[g], m.r))
    axes = []
    for a, d, size in found:
        a = a - (a @ d) * d
        for k, (b, e, reach) in enumerate(axes):
            if abs(d @ e) > math.cos(math.radians(0.5)) and np.linalg.norm(np.cross(a - b, e)) < TOL * 4:
                axes[k] = (b, e, max(reach, 2 * size + 3))
                break
        else:
            axes.append((a, d, 2 * size + 3))
    return axes


def _axis_pass(mesh, regions, axes, tol=None):
    """Patches on the given axes (cylinders, cones, tori) among the free facets, held to
    `tol` if given (else the mesh's own tolerance)."""
    global _mesh_tol, _axis_vouched
    strict = _mesh_tol
    if not tol or tol <= strict:
        return _axis_patches(mesh, regions, axes)
    # the wider allowance only for straight profiles (cylinders, cones: only the
    # radius and slope free); a torus's three free numbers find false fits in it
    _mesh_tol, _axis_vouched = tol, True
    try:
        out = _axis_patches(mesh, regions, axes, lines_only=True)
    finally:
        _mesh_tol, _axis_vouched = strict, False
    return out + _axis_patches(mesh, regions, axes)


def _near(axes, pts, most=6):
    """Only the axes that pass close enough to these points to matter (the nearest few)."""
    if not axes:
        return []
    key = id(axes)
    if _near.cache.get("key") != key:
        _near.cache = {"key": key, "A": np.array([a for a, _, _ in axes]),
                       "D": np.array([d for _, d, _ in axes]), "R": np.array([r for _, _, r in axes]),
                       "axes": axes}
    c = _near.cache
    dist = np.linalg.norm(np.cross(pts.mean(axis=0) - c["A"], c["D"]), axis=1)
    ok = np.nonzero(dist <= c["R"])[0]
    ok = ok[np.argsort(dist[ok])][:most]
    return [(c["A"][k], c["D"][k]) for k in ok]


_near.cache = {}


# ---------------------------------------------------------------- features

def _arc(angles, min_points=6):
    """(start, span) of the arc covered by a set of angles; span 2*pi for a full ring."""
    a = np.sort(np.mod(angles, TWO_PI))
    if len(a) < 2:
        return None
    a = a[np.r_[True, np.diff(a) > 1e-6]]   # merge repeats (the same angle on several rings)
    if len(a) < 2:
        return None
    gaps = np.diff(np.r_[a, a[0] + TWO_PI])
    g = gaps.argmax()
    if len(a) >= min_points and gaps[g] <= math.radians(SMOOTH_DEG + 5):
        return 0.0, TWO_PI
    return a[(g + 1) % len(a)], TWO_PI - gaps[g]


def _angle_gap(a, b):
    return np.abs(np.mod(a - b + math.pi, TWO_PI) - math.pi)


@dataclass
class Feature:
    model: object
    label: str          # what it is, in plain words
    detail: str
    convex: bool
    kind: str           # "revolve" (cylinder/cone/torus/sphere cap), "wedge" (sphere corner),
                        # "closed" (a whole closed surface with no outline: a ball, a ring),
                        # "trimmed" (any of these surfaces, cut to an arbitrary outline)
                        # or "blend" (a smooth freeform face where no simple surface fits)
    facets: np.ndarray  # mesh facets this patch replaces
    change: float       # volume the exact surface adds compared with the facets (mm^3)
    tolerance: float    # how far the actual change may differ from that
    worst: float        # largest gap between a facet and the true surface (mm)
    u0: float = 0.0     # "revolve": angular range around the axis
    span: float = TWO_PI
    lo: float = 0.0     # "revolve": profile range (height for a line, tube angle for a circle)
    hi: float = 0.0
    planes: tuple = ()  # "wedge": planes through the centre bounding the corner
    parts: tuple = ()   # "blend": the pieces it replaced (put back if it can't be built)
    depth: int = 0      # "blend": how many times it has been cut in two to make it fit

    def describe(self):
        return f"{self.label:<22} {self.detail}"


def _feature(mesh, model, fids, convex):
    tids = np.concatenate([mesh.ftris[f] for f in fids])
    T = mesh.tris[tids]
    P = mesh.pts[_uniq(T.ravel())]
    uniq, counts = _edge_counts(T, len(mesh.pts))
    rim_edges = uniq[counts == 1]
    rim = mesh.pts[_uniq(rim_edges.ravel())]

    # How far each facet sits from the true surface, sampled at its edge midpoints
    # (exact for the near-parabolic gap between a flat facet and a curved surface).
    corners = mesh.pts[T]
    s = model.signed(((corners + corners[:, [1, 2, 0]]) / 2).reshape(-1, 3)).reshape(-1, 3)
    area = mesh.tarea[tids]
    change = (-1 if convex else 1) * float((area * s.mean(axis=1)).sum())
    gain = float((area * np.abs(s).mean(axis=1)).sum())
    worst = float(np.abs(s).max())
    if worst < 1e-5:
        return None
    if _loose and not _anchored and (worst > LOOSE_MAX_GAP or not _loose_support(mesh, fids, model.kind)):
        return None
    base = dict(convex=convex, change=change, tolerance=0.5 * gain + 1e-3, facets=np.asarray(fids),
                worst=worst)

    if isinstance(model, Sphere):
        if len(rim) == 0:
            label = "ball" if convex else "spherical cavity"
            return Feature(model, label, f"dia {2 * model.r:.3f}", kind="closed", **base)
        mid = rim.mean(axis=0)
        n = _right_vectors(rim - mid)[2]
        if np.abs((rim - mid) @ n).max() <= _tol(model):
            # Flat circular edge: a dome, dimple or hemispherical tip, handled as a revolved cap.
            if ((P - mid) @ n).mean() < 0:
                n = -n
            cap = Revolved(model.c, n, circle=(0.0, float(model.c @ n), model.r))
            return _revolved_feature(mesh, cap, P, rim, rim_edges, base)
        directions = (P - model.c) / model.r
        spread = math.acos(max(-1.0, min(1.0, (directions @ directions.mean(axis=0)).min()
                                         / np.linalg.norm(directions.mean(axis=0)))))
        if not _loose and (len(P) < MIN_CORNERS["sphere"]
                           or 2 * spread < math.radians(MIN_SPAN_DEG)):
            return None
        planes = _great_circles(model, P, mesh.pts, rim_edges)
        if planes is None or len(planes) > 4:
            if not _loose and not _enough_for_trimmed(mesh, fids, P, rim_edges):
                return None
            label = "rounded area (sphere)" if convex else "spherical hollow"
            return Feature(model, label, f"sphere r {model.r:.3f}, cut to shape", kind="trimmed", **base)
        return Feature(model, "rounded corner", f"sphere r {model.r:.3f}", kind="wedge",
                       planes=tuple(planes), **base)
    return _revolved_feature(mesh, model, P, rim, rim_edges, base)


def _loose_support(mesh, fids, kind):
    """stlToSolid-style minimum evidence: enough facets, turning through enough angle."""
    n = mesh.fn[fids]
    if _fewer_distinct(np.round(n, 3), LOOSE_MIN_FACETS[kind]):
        return False
    return _least_pair_dot(n, math.cos(math.radians(LOOSE_MIN_TURN_DEG[kind]))) <= math.cos(math.radians(LOOSE_MIN_TURN_DEG[kind]))


def _enough_for_trimmed(mesh, fids, P, free_edges):
    """Is a patch with an outline off its natural edges believable?

    Where it leaves those edges it must meet something else at a real crease (a hole
    breaking out through a sloped face, a knob cut by another feature). A smooth
    hand-over there means it's more likely a slice of some larger curved surface
    (one ring of a torus, say, which lies exactly on a sphere).
    """
    if len(P) < TRIMMED_MIN_CORNERS or len(fids) < TRIMMED_MIN_FACETS:
        return False
    n = mesh.fn[fids]
    if _least_pair_dot(n, math.cos(math.radians(TRIMMED_MIN_TURN_DEG))) > math.cos(math.radians(TRIMMED_MIN_TURN_DEG)):
        return False
    # (Mostly, anyway: two equal holes crossing are tangent at a couple of points.)
    inside = set(np.asarray(fids).tolist())
    crease = math.cos(math.radians(CREASE_DEG))
    smooth = total = 0.0
    for a, b in free_edges:
        sides = mesh.edge_facets.get((min(a, b), max(a, b)), [])
        mine = [f for f in sides if f in inside]
        other = [f for f in sides if f not in inside]
        length = float(np.linalg.norm(mesh.pts[a] - mesh.pts[b]))
        total += length
        if mine and other and mesh.fn[mine[0]] @ mesh.fn[other[0]] > crease:
            smooth += length
    return smooth <= 0.25 * total


def _great_circles(model, P, pts, rim_edges):
    """Planes through the sphere centre that together make up a patch's outline."""
    tol = _tol(model)
    normals = []
    for i, j in rim_edges:
        m = np.cross(pts[i] - model.c, pts[j] - model.c)
        if np.linalg.norm(m) < 1e-9:
            continue
        m /= np.linalg.norm(m)
        if not any(abs(m @ q) > math.cos(math.radians(1)) for q in normals):
            normals.append(m)
            if len(normals) > 8:
                # (no simple corner, as below: said at once, not after comparing every
                # outline edge with every plane so far, minutes on a big sphere patch)
                return None
    if not 1 <= len(normals) <= 8:
        return None
    # both ends of every outline edge must lie on the same plane
    on = np.abs((pts[rim_edges] - model.c) @ np.array(normals).T) <= tol
    if not (on[:, 0] & on[:, 1]).any(axis=1).all():
        return None
    planes = []
    for m in normals:
        side = (P - model.c) @ m
        if side.min() >= -tol:
            planes.append(m)
        elif side.max() <= tol:
            planes.append(-m)
        else:
            return None                 # patch is on both sides: not a simple corner
    return planes


def _revolved_feature(mesh, model, P, rim, rim_edges, base):
    tol = _tol(model)
    rho, z, w = model.local(P)
    off_axis = rho > tol
    around = _arc(model.angle(w[off_axis]))
    if around is None:
        return None
    u0, span = around
    r_rho, r_z, r_w = model.local(rim) if len(rim) else (np.zeros(0),) * 3
    r_u = model.angle(r_w) if len(rim) else np.zeros(0)

    if model.line:
        lo, hi = z.min(), z.max()
        at = lambda end: np.abs(r_z - end) <= tol
        c0, k = model.line
        if k != 0 and span >= TWO_PI:
            # A pointed cone tip can be closed off by one flat facet with no corner at
            # the tip itself; if the outline doesn't reach that end, run to the tip.
            apex = -c0 / k
            if abs(apex - lo) < abs(apex - hi) and not at(lo).any():
                lo = apex
            elif abs(apex - hi) < abs(apex - lo) and not at(hi).any():
                hi = apex
    else:
        rc, zc, r = model.circle
        ring = _arc(np.arctan2(z - zc, rho - rc))
        if ring is None:
            return None
        if ring[1] >= TWO_PI - 1e-9 and (span >= TWO_PI or rc - r <= tol):
            if span >= TWO_PI and rc - r > tol and not len(rim_edges):
                # a whole ring (an O-ring, a torus on its own): a closed surface, no outline
                # (refused, its pieces were left, and locked in the merged engine they kept
                # MMM from making the one face it is: cad_torus 392 faces, now 1)
                label = "ring" if base["convex"] else "ring-shaped cavity"
                return Feature(model, label, f"torus r {r:.3f} round a {2 * rc:.3f} circle", kind="closed",
                               u0=0.0, span=TWO_PI, lo=0.0, hi=TWO_PI, **base)
            return None                 # a ring cut to shape, or a tube crossing itself
        # (a tube all round but only part way round the axis is a bent pipe's bore)
        lo, hi = ring[0], ring[0] + ring[1]
        v = np.arctan2(r_z - zc, r_rho - rc)
        at = lambda end: _angle_gap(v, end) * r <= tol
        if rc == 0 and span >= TWO_PI:
            # Same for the pole of a dome or dimple closed off by one flat facet.
            lo, hi = math.remainder(lo, TWO_PI), math.remainder(lo, TWO_PI) + (hi - lo)
            if not at(hi).any() and hi > 0:
                hi = math.pi / 2
            if not at(lo).any() and lo < 0:
                lo = -math.pi / 2

    # Every outline edge must run along one of the patch's natural boundary lines
    # (both of its ends on the same line), so the outline traces the patch exactly.
    lines = [at(lo), at(hi)]
    if span < TWO_PI:
        on_axis = r_rho <= tol
        lines += [(_angle_gap(r_u, u0) * r_rho <= tol) | on_axis,
                  (_angle_gap(r_u, u0 + span) * r_rho <= tol) | on_axis]
    trimmed = False
    if len(rim_edges):
        index = {v: i for i, v in enumerate(np.unique(rim_edges))}
        a = np.array([index[i] for i in rim_edges[:, 0]])
        b = np.array([index[j] for j in rim_edges[:, 1]])
        along = np.any([line[a] & line[b] for line in lines], axis=0)
        if not along.all():
            # the outline leaves the natural edges: only OK as a well-supported trimmed patch
            vouched = _axis_vouched and len(base["facets"]) >= TRIMMED_MIN_FACETS
            if not _loose and not vouched and not _enough_for_trimmed(mesh, base["facets"], P, rim_edges[~along]):
                return None
            trimmed = True
    if not _loose:
        if len(P) < MIN_CORNERS[model.kind]:
            return None                 # too few points to be sure of the shape
        min_span = math.radians(MIN_SPAN_DEG)
        if span < min_span or (not model.line and hi - lo < min_span):
            return None
        if model.kind == "cone" and not _creased(mesh, base["facets"], z, lo, hi, model, tol):
            return None

    if model.circle and any(model.circle[0] + model.circle[2] * math.cos(v) < -tol
                            for v in np.linspace(lo, hi, 9)):
        return None                     # profile would cross the axis
    label, detail = _describe(model, base["convex"], span, lo, hi)
    if trimmed:
        detail += ", cut to shape"
    return Feature(model, label, detail, kind="trimmed" if trimmed else "revolve",
                   u0=u0, span=span, lo=lo, hi=hi, **base)


def _creased(mesh, fids, z, lo, hi, model, tol):
    """Does a one-ring cone meet a neighbour at a real crease?

    A single ring of facets on any rounded surface is exactly a slice of a cone, so
    a one-ring cone is only trusted if at least one of its end circles is a crease
    (or an open edge) rather than a smooth continuation.
    """
    if ((np.abs(z - lo) > tol) & (np.abs(z - hi) > tol)).any():
        return True                     # several rings of facets: a genuine cone
    inside = set(fids.tolist())
    soft = set()
    for f in fids:
        for g in mesh.nbrs[f]:
            if g in inside:
                continue
            if math.degrees(math.acos(min(1.0, mesh.fn[f] @ mesh.fn[g]))) >= CREASE_DEG:
                continue
            for a, b in mesh.shared[(min(f, g), max(f, g))]:
                ez = model.local(mesh.pts[[a, b]])[1]
                for end in (lo, hi):
                    if (np.abs(ez - end) <= tol).all():
                        soft.add(end)
    c0, k = model.line
    ends = {end for end in (lo, hi) if c0 + k * end > tol}   # a pointed tip has no edge
    return not ends <= soft


def _describe(model, convex, span, lo, hi):
    full = span >= TWO_PI
    kind = model.kind
    if kind == "cylinder":
        r = model.line[0]
        if full:
            return ("pin" if convex else "hole"), f"dia {2 * r:.3f}, length {hi - lo:.2f}"
        return "rounded edge", f"r {r:.3f}, {math.degrees(span):.0f} deg, length {hi - lo:.2f}"
    if kind == "cone":
        c0, k = model.line
        angle = math.degrees(math.atan(abs(k)))
        dias = f"dia {2 * (c0 + k * lo):.3f} -> {2 * (c0 + k * hi):.3f}"
        if full:
            return ("cone / chamfered pin" if convex else "countersink / chamfer"), f"{dias}, {angle:.1f} deg"
        return "chamfer (cone)", f"{dias}, {angle:.1f} deg, {math.degrees(span):.0f} deg around"
    if kind == "torus":
        rc, zc, r = model.circle
        return "rounded edge (curved)", f"r {r:.3f} around a {2 * rc:.3f} circle, {math.degrees(span):.0f} deg"
    r = model.circle[2]
    height = r * (math.sin(hi) - math.sin(lo))
    return ("dome" if convex else "dimple"), f"sphere dia {2 * r:.3f}, height {height:.2f}"


def summarize(features):
    counts = Counter(f.label for f in features)
    return ", ".join(f"{label} x{n}" for label, n in counts.most_common())
