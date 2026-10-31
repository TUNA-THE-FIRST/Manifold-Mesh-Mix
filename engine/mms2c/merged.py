"""
The merged engine: stl2curves and Manifold Mesh Mix (MMM, the mmm package, AGPL-3.0) as one
pipeline, meeting on one labelling of each shell's triangles.

stl2curves' front end reads and mends the mesh (repair, thinning, bodies split where they
touch) and finds what MMM can't: screw threads, rolling-ball fillets, radii snapped to
the design size, freeform patches, pipes along curved edges. Each feature becomes a
region of MMM's Segmenter whose surface is locked: MMM may move its border but never
refits it. MMM's own seeds then cover only the triangles we left, and its border
clean-up (teeth, min-cut straightening, slivers) runs over all of them, ours included.
From there MMM's pipeline carries on: the B-rep is assembled without sewing (each border
one fitted edge shared by both faces), faces that won't build or stray from the mesh fall
back face by face (freeform, then flat), every result is judged against the mesh, organic
bodies get smooth surfaces, and the STEP file is written, read back and repaired.

Surfaces cross over as MMM's own types where it has them (planes, cylinders, cones,
spheres, tori: our snapped values) and otherwise wrapped (OurSurface: a thread piece,
a freeform patch, a pipe, an extrusion, with the OCC surface our builder would cut).
Our N-sided blends are left to MMM, which fits its own freeform cells there.

Meanwhile stl2curves alone and MMM alone convert the same file in processes of their
own, and the best of the three finished results is saved (rivals.py).

Every file passes one entrance first (convert_file -> _entrance): the paths are checked
(pipeline.check_paths), the mesh is read by MMM's reader and given its refusals
(meshio.check_triangles), so a broken file ends in a message for the user, not a
traceback from deep inside a stage. OBJ, PLY and 3MF files go to MMM's pipeline alone,
as in the app; and if our front end fails on a file that passed, MMM's own reading and
mending take over (front), with a warning saying so.
"""
import math
import os
import shutil
import tempfile
import time

import numpy as np

from mmm import cad as mcad
from mmm import fitting as mfit
from mmm import meshio as mio
from mmm import pipeline as mpipe
from mmm.report import NULL, Reporter

from . import build, convert, rivals, stages, workers
from . import features as F

# the app's Auto quality by triangle count (MMM desktop/mmm_app/settings.py)
AUTO_QUALITY_STEPS = ((50_000, 0.9), (300_000, 0.8), (1_000_000, 0.7), (3_000_000, 0.6))
SPINDLE = 1.0   # a torus whose centre circle is no bigger than this times its tube stays ours
# stl2curves alone and MMM alone convert the file alongside, as rivals (rivals.py), only
# with MM_S2C_RIVALS=1 (comparisons): off by default since 2026-10-29, as the owners
# decided for the test builds, so the engine is one pipeline that picks its methods from
# the shape
RIVALS = os.environ.get("MM_S2C_RIVALS") == "1"
# our smooth blends, freeform patches and pipes as seeds too (add_blends: on the cam and
# tube MMM test models most of the front end's time); MM_S2C_FRONT_BLENDS=0 seeds with
# the exact features only, and our own conversion alongside fits the blends itself.
# Measured 2026-10-25 (28-part sample, no rivals): without them 10,483 faces instead of
# 7,934 (+32%; test13 106 -> 407, the hook 154 -> 316) for 5% less time: keep them
FRONT_BLENDS = os.environ.get("MM_S2C_FRONT_BLENDS") != "0"
# formats MMM's reader takes on its own (everything else is read as an STL, as before):
# these go to MMM's pipeline without our front end
OTHER_FORMATS = (".obj", ".ply", ".off")
# 3MF projects go through our front end like an STL (owners' choice, 2026-10-30): before,
# they took MMM's path alone and lost every feature (Bravo's skittles nut, a thread, came
# out as 3,272 freeform faces). The plate is read with our 3MF reader, every object placed
# and scaled as printed, so the STEP is laid out as on the plate (also the owners' choice;
# MMM's reader gave each part in its own frame, and a 13-part plate came out in a pile)
PLATE_FORMATS = (".3mf",)
# model units across (mm unless chosen otherwise) our front end works within: smaller,
# its 1e-4 mm weld crushes the shape (a 2 mm sphere came out in 16 faces); larger, its
# thread search's 0.02 mm bands run away (a 1e8 mm sphere asked for 303 GiB). A model
# outside goes to MMM's pipeline alone, which welds relative to the size
FRONT_SIZES = (0.1, 1e5)
# sizes that suggest the wrong unit: MMM's mesh check's note (meshcheck step 13), given
# here when our front end reads the file instead
UNIT_NOTE = (1.0, 3000.0)
FLAT = 1e-9         # a closed body enclosing less than this times the model's diagonal cubed: no volume
INSIDE_OWN = 0.99   # share of a feature's own corners that must pass its surface's domain test to use it
# our seeds against MMM's own segmentation where they are unsure (seeded_segmenter):
PATCH_SEEDS = 3         # seeds of ours (threads aside) in one smooth area that call for the comparison
PATCH_CREASE_DEG = 30.0  # deg: a smooth area ends where triangles meet at more than this (as MMM's _find_sweeps)
SURE_SHARE = 0.99       # share of a seed's triangles MMM's own accept test must take for the seed to be sure
EXACT_KINDS = ("plane", "cylinder", "cone", "sphere", "torus")     # the surfaces a designer draws
FREEFORM_SAVES = 0.75   # a way using freeform surfaces beats an all-exact one only with at most this share of its regions


class OurSurface:
    """One of our surface models as an MMM region surface ("freeform" to MMM: it builds
    the face on geom() and measures with distance()/normal()). normal() follows the OCC
    surface's own orientation, as MMM's freeform cells do, so a region's sense says which
    way the face is turned.

    distance() measures the surface the face is built on: our model's own distance where
    the point lies over geom()'s domain (inside, see _inside), and none at all (infinite)
    beyond it. The model alone is valid anywhere (a freeform height field extends its
    spline, a pipe measures to its end's ball), but geom() covers only the fitted patch and
    a margin: on the GPS case back a flat triangle of 119 mm2 lay within 0.1 mm of a
    freeform patch's model but 39 mm off its surface, MMM took it into that region, the
    face couldn't cover it, and the whole body (518 faces) was saved as 11,690 triangles."""

    kind = "freeform"
    rank = 5
    locked = True
    dev = 0.0

    def __init__(self, model, geom, flip, inside=None):
        self.model, self._geom, self.flip = model, geom, flip
        self.inside = inside

    def distance(self, P):
        P = np.atleast_2d(P)
        d = np.abs(self.model.signed(P))
        if self.inside is None:
            return d
        return np.where(self.inside(P), d, np.inf)

    def normal(self, P):
        return self.flip * self.model.normal(np.atleast_2d(P))

    def degenerate(self, ext):
        return False

    def same_as(self, o, tol, ext):
        return o is self

    def describe(self):
        return f"stl2curves {self.model.kind}"

    def geom(self):
        return self._geom


def quality(n):
    return next((q for top, q in AUTO_QUALITY_STEPS if n < top), 0.5)


def mmm_surface(feature, mesh):
    """The MMM surface for one of our features, locked; None to leave its triangles to
    MMM (an N-sided blend, a surface that can't be made)."""
    m = feature.model
    if feature.kind == "blend" or m is None:
        return None
    prim = None
    if isinstance(m, F.Sphere):
        prim = mfit.Sphere(m.c, m.r)
    elif isinstance(m, F.Revolved):
        a, d = np.asarray(m.a, float), np.asarray(m.d, float)
        if m.line:
            c0, k = m.line
            if k == 0:
                prim = mfit.Cylinder(a, d, c0)
            else:
                z = -c0 / k              # the apex, where the profile meets the axis
                # (MMM's cone opens along d from its apex: up the axis when the radius grows)
                prim = mfit.Cone(a + z * d, d if k > 0 else -d, math.atan(abs(k)))
        else:
            rc, zc, r = m.circle
            centre = a + zc * d
            if rc == 0:
                prim = mfit.Sphere(centre, r)
            elif rc > SPINDLE * r:
                prim = mfit.Torus(centre, d, rc, r)
    if prim is not None:
        prim.locked = True
        return prim
    geom = _occ_surface(feature, mesh)
    if geom is None:
        return None
    return OurSurface(m, geom, _orientation(m, geom, mesh, feature), _inside(feature, mesh))


def _inside(feature, mesh):
    """The test whether points lie over the domain of the OCC surface this feature's face
    is built on (_occ_surface), in the model's own parameters; None (no test) when the
    feature's own corners don't pass it (INSIDE_OWN): then the test, not the surface, is
    wrong, and the model's distance stands as before."""
    m = feature.model
    if m.kind == "thread":
        (u0, u1), (lo, hi) = build.thread_reach(feature)

        def test(P):
            s, _, U = m.place(P)
            return (s >= lo) & (s <= hi) & (U >= u0) & (U <= u1)
    elif hasattr(m, "inside"):
        test = m.inside
    else:
        return None
    tids = np.concatenate([mesh.ftris[f] for f in feature.facets])
    own = mesh.pts[np.unique(mesh.tris[tids])]
    try:
        if test(own).mean() < INSIDE_OWN:
            return None
    except Exception:
        return None
    return test


def _occ_surface(feature, mesh):
    """The OCC surface our builder cuts this feature's face from (None if it has none)."""
    from OCP.BRep import BRep_Tool
    m = feature.model
    try:
        if m.kind == "thread":
            return build._thread_surface(feature, m)
        if m.kind in ("freeform", "pipe", "extrusion"):
            face = m.face()
            return None if face is None else BRep_Tool.Surface_s(face)
    except Exception:
        return None
    return None


def _fit_surfaces(parts):
    """Fit the OCC surfaces of our freeform patches, pipes and extrusions in the worker
    processes, all at once, before the handover asks for them one by one (test12: 20 s
    of its 22.5 on one core). Each comes back in a BRep file (build._surface_to /
    _surface_from, as our builder hands them about); any that fails is fitted here."""
    models = [f.model for mesh, features, _ in parts for f in features
              if f.kind != "blend" and getattr(f.model, "kind", None) in ("freeform", "pipe", "extrusion")
              and getattr(f.model, "_surface", 0) is None]
    pool = workers.get() if len(models) > 1 else None
    if pool is None:
        return
    tmp = tempfile.mkdtemp(prefix="s2c_surfaces_")
    try:
        paths = [os.path.join(tmp, f"{i}.brep") for i in range(len(models))]
        futures = [pool.submit(build._surface_to, m, p) for m, p in zip(models, paths)]
        for m, p, future in zip(models, paths, futures):
            if future.exception() is None:
                build._surface_from(m, p)
    except Exception:
        workers.broken()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _orientation(model, geom, mesh, feature, n=24):
    """+1 when the model's normal() points the way the OCC surface's own normal does,
    else -1 (measured at a few of the feature's corners)."""
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.GeomLProp import GeomLProp_SLProps
    from OCP.gp import gp_Pnt
    tids = np.concatenate([mesh.ftris[f] for f in feature.facets])
    P = mesh.pts[np.unique(mesh.tris[tids])]
    P = P[np.linspace(0, len(P) - 1, min(n, len(P))).astype(int)]
    N = model.normal(P)
    votes = 0.0
    for p, nm in zip(P, N):
        pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, p)), geom)
        if not pr.NbPoints():
            continue
        u, v = pr.LowerDistanceParameters()
        props = GeomLProp_SLProps(geom, u, v, 1, 1e-9)
        if props.IsNormalDefined():
            votes += float(np.dot(nm, props.Normal().Coord()))
    return 1.0 if votes >= 0 else -1.0


# ----------------------------------------------------------------------------
# Our bodies as MMM shells, our features as their regions
# ----------------------------------------------------------------------------


class _Body:
    """One body for MMM: its shells (outer first, then cavities), and for each shell
    the regions our features give it: [(surface, shell triangle numbers)]; part: which
    of our prepared parts it came from, alone: whether it is all of that part; pieces:
    which of the part's pieces (_shells_of) each shell is."""

    def __init__(self):
        self.shells, self.seeds, self.pieces = [], [], []
        self.part, self.alone = None, False


def _split_pinches(V, F):
    """Where more than two triangles share an edge (two parts of a body touching along
    it), each vertex gets one copy per fan of triangles joined through two-sided edges,
    so every edge is two-sided again and the shell closed, as MMM's own repair leaves it
    (ours keeps such edges: on a GPS case back, one edge 0.6 mm long shared by four
    triangles left MMM's shell open, and the merged result was saved as surfaces).
    Triangle rows stay as they are, so our features still map onto them."""
    topo = mio.EdgeTopology(V, F)
    if not topo.nonmanifold_he.any():
        return V, F
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    Fr = F.reshape(-1)
    h = np.nonzero(topo.twin >= 0)[0]
    t = topo.twin[h]
    h, t = h[h < t], t[h < t]

    def nxt(x):
        return 3 * (x // 3) + (x % 3 + 1) % 3

    # a shared edge's triangles pair up around it: each with its neighbour by angle across
    # a wedge of material (the side away from both their normals), the two parts touching
    # there kept apart. The pairs count as two-sided edges below, so the fans at both ends
    # of the edge agree (fans found at each end alone paired them crosswise on that case
    # back, which left the edge open)
    pairs = []
    nm = np.nonzero(topo.nonmanifold_he)[0]
    ends = np.stack([Fr[nm], Fr[nxt(nm)]], axis=1)
    _, group = np.unique(np.sort(ends, axis=1), axis=0, return_inverse=True)
    N = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    for g in range(group.max() + 1):
        hs = nm[group.reshape(-1) == g]
        if len(hs) % 2:
            continue
        p0 = V[Fr[hs[0]]]
        d = V[Fr[nxt(hs[0])]] - p0
        d = d / max(np.linalg.norm(d), 1e-30)
        u = np.cross(d, [1.0, 0.0, 0.0] if abs(d[0]) < 0.9 else [0.0, 1.0, 0.0])
        u /= np.linalg.norm(u)
        w = np.cross(d, u)
        opp = V[Fr[nxt(nxt(hs))]] - p0
        ang = np.arctan2(opp @ w, opp @ u)
        order = np.argsort(ang)
        hs, ang = hs[order], ang[order]
        best = None
        for shift in (0, 1):
            ks = [(k + shift) % len(hs) for k in range(0, len(hs), 2)]
            material = 0
            for k in ks:
                k2 = (k + 1) % len(hs)
                a0, a1 = ang[k], ang[k2] + (2 * np.pi if k2 <= k else 0.0)
                mid = 0.5 * (a0 + a1)
                m = np.cos(mid) * u + np.sin(mid) * w
                material += int(N[hs[k] // 3] @ m < 0) + int(N[hs[k2] // 3] @ m < 0)
            if best is None or material > best[0]:
                best = (material, ks)
        pairs += [(hs[k], hs[(k + 1) % len(hs)]) for k in best[1]]
    if pairs:
        pairs = np.array(pairs, dtype=np.int64)
        h, t = np.concatenate([h, pairs[:, 0]]), np.concatenate([t, pairs[:, 1]])
    # corner 3f+k is vertex F[f, k] in triangle f; a two-sided edge joins the corners
    # of its two vertices in its two triangles
    same = Fr[h] == Fr[t]
    i = np.concatenate([h, nxt(h)])
    j = np.concatenate([np.where(same, t, nxt(t)), np.where(same, nxt(t), t)])
    n = 3 * len(F)
    _, comp = connected_components(coo_matrix((np.ones(len(i), dtype=np.int8), (i, j)), shape=(n, n)),
                                   directed=False)
    key = Fr.astype(np.int64) * n + comp
    uk, inv = np.unique(key, return_inverse=True)
    return V[uk // n], inv.reshape(-1, 3)


def _shells_of(pts, tris):
    """The connected pieces of one of our bodies as MMM shells, each with the rows of
    tris it holds (in order, so triangle numbers map across)."""
    V, Fc = _split_pinches(*mio.compact(pts, tris))
    topo = mio.EdgeTopology(V, Fc)
    n, lab = mio.face_components(Fc, topo)
    out = []
    for c in range(n):
        rows = np.nonzero(lab == c)[0]
        Vs, Fs = mio.compact(V, Fc[rows])
        out.append((mio.Shell(Vs, Fs), rows))
    return out


def bodies_from(parts, groups, pts, pieces_of=None):
    """MMM bodies from our prepared parts, each with, for its shells, the locked regions
    our features give each (_Body.seeds). pieces_of: each part's pieces (_shells_of), if
    made already (route)."""
    bodies = []
    biggest = max(_area(pts, g) for g in groups)
    for gi, ((mesh, features, _), g) in enumerate(zip(parts, groups)):
        pieces = pieces_of[gi] if pieces_of is not None else _shells_of(pts, g)
        seeds = part_seeds(mesh, features, pts, g, pieces)
        made = _nest([p[0] for p in pieces], seeds)
        made = [b for b in made if not _debris(b.shells[0], biggest)]
        for b in made:
            # (our own build of this part is a candidate for the body only when the
            # part is that one body)
            b.part, b.alone = gi, len(made) == 1
        bodies += made
    return bodies


def part_seeds(mesh, features, pts, g, pieces):
    """For each of a part's pieces (_shells_of), the locked regions our features give it:
    [(surface, piece triangle numbers)]."""
    # (our Mesh drops zero-area triangles: theirs map back through this)
    a, b, c = (pts[g[:, k]] for k in range(3))
    kept = np.nonzero(np.linalg.norm(np.cross(b - a, c - a), axis=1) > 1e-12)[0]
    if len(kept) != len(mesh.tris) or not np.array_equal(g[kept], mesh.tris):
        raise RuntimeError("our mesh's triangles don't map onto the body's")
    where = np.empty((len(g), 2), dtype=np.int64)       # body row -> (piece, triangle)
    for i, (_, rows) in enumerate(pieces):
        where[rows, 0] = i
        where[rows, 1] = np.arange(len(rows))
    seeds = [[] for _ in pieces]
    for feature in features:
        surf = mmm_surface(feature, mesh)
        if surf is None:
            continue
        tids = kept[np.concatenate([mesh.ftris[f] for f in feature.facets])]
        for i in np.unique(where[tids, 0]):
            seeds[i].append((surf, where[tids[where[tids, 0] == i], 1]))
    return seeds


def _area(pts, tris):
    a, b, c = (pts[tris[:, k]] for k in range(3))
    return float(np.linalg.norm(np.cross(b - a, c - a), axis=1).sum() / 2)


def _debris(shell, biggest):
    """A loose piece MMM's reader drops (meshcheck.prepare_part): flat (no volume to
    speak of), or tiny beside the biggest piece. Rak N-USB has one of 12 triangles that
    no method could build, and its faceted copy cost faces (the biggest always stays)."""
    a = _area(shell.V, shell.F)
    if a >= biggest:
        return False
    flat = shell.closed and abs(shell.volume) < 1e-9 * max(a, 1e-30) ** 1.5
    return (len(shell.F) < 4 and a < 0.01 * biggest) or flat or a < 5e-5 * biggest


def _nest(shells, seeds):
    """Shells into bodies, as MMM's organise_bodies does: a shell inside no other is a
    body's outer shell (turned outward if it faces in); one inside another is a cavity
    of the smallest one around it if it faces the other way, else a body of its own.
    (Turning a shell round keeps its triangles' numbers, so the seeds still fit.)"""
    box = [(s.V.min(0), s.V.max(0)) for s in shells]

    def inside(j, i):
        return i != j and (box[i][0] <= box[j][0]).all() and (box[i][1] >= box[j][1]).all() \
            and abs(shells[i].volume) > abs(shells[j].volume)

    around = {j: [i for i in range(len(shells)) if inside(j, i)] for j in range(len(shells))}
    for j, s in enumerate(shells):
        if s.closed and not around[j] and s.volume < 0:
            shells[j] = mio.Shell(s.V, s.F[:, ::-1].copy())
    home = {}
    for j, s in enumerate(shells):
        if s.closed and around[j]:
            i = min(around[j], key=lambda i: abs(shells[i].volume))
            if (shells[i].volume > 0) != (s.volume > 0):
                home[j] = i
    bodies = {}
    for j in range(len(shells)):
        if j not in home:
            bodies[j] = _Body()
    for j in range(len(shells)):
        i = j
        while i in home:
            i = home[i]
        bodies[i].shells.append(shells[j])
        bodies[i].seeds.append(seeds[j])
        bodies[i].pieces.append(j)
    out = []
    for i, b in bodies.items():
        # (the outer shell first)
        k = b.shells.index(shells[i])
        b.shells.insert(0, b.shells.pop(k))
        b.seeds.insert(0, b.seeds.pop(k))
        b.pieces.insert(0, b.pieces.pop(k))
        out.append(b)
    return out


def seeded_segmenter(shell, seeds, tol, quality, report=NULL):
    """MMM's Segmenter for one shell, seeded with our features (_segment), checked where
    they are unsure against MMM's own segmentation (on demand).

    Our front end fits each piece within its own limit (0.1 mm by default) and commits
    greedily, with no step that merges pieces up; MMM fits to a tighter tolerance and
    merges the pieces of one fillet or sweep into one face, but shatters narrow bands we
    find whole. Locked, our pieces of a curving fillet kept MMM from its one face
    (cad_tube_fillet 30 faces where MMM alone makes 15, the cam 22 where it makes 8);
    offered unlocked they still steered it (36 and 77); dropped wholesale, MMM shattered
    the countersinks we had right (Front Lid 53 -> 134). So a smooth area (bounded by
    creases sharper than PATCH_CREASE_DEG) holding PATCH_SEEDS or more of our seeds,
    threads aside, one of them unsure (_sure), is segmented both ways, with our seeds and
    by MMM alone (those seeds left out), and the way _winner picks is kept there. Where
    nothing is unsure, it is as before. (On 17 parts: tube 30 -> 15, cam 22 -> 8, hook
    154 -> 109, liner 141 -> 93; Front Lid, the 608 bearing, test13 unchanged.)"""
    held = _Held(report)
    A = _segment(shell, seeds, tol, quality, held)
    comp = _smooth_areas(A)
    where = [int(np.bincount(comp[np.asarray(f, dtype=np.int64)]).argmax()) for _, f in seeds]
    count, unsure = {}, set()
    for i, (surf, faces) in enumerate(seeds):
        if _thread(surf):
            continue
        a = where[i]
        count[a] = count.get(a, 0) + 1
        if a not in unsure and not _sure(A, surf, faces):
            unsure.add(a)
    areas = {a for a in unsure if count[a] >= PATCH_SEEDS}
    if not areas:
        held.replay()
        return A
    mine = {i for i, (surf, _) in enumerate(seeds) if where[i] in areas and not _thread(surf)}
    held_b = _Held(report)
    B = _segment(shell, [s for i, s in enumerate(seeds) if i not in mine], tol, quality, held_b)
    won = {a for a in areas if _winner(_census(A, comp, a), _census(B, comp, a)) == "mmm"}
    report.log(f"In {len(areas)} smooth area(s) stl2curves' pieces were unsure, so MMM's own segmentation was "
               f"tried there as well: kept in {len(won)}, stl2curves' in {len(areas) - len(won)}")
    if not won:
        held.replay()
        return A
    if won == areas:
        held_b.replay()
        return B
    drop = {i for i in mine if where[i] in won}
    return _segment(shell, [s for i, s in enumerate(seeds) if i not in drop], tol, quality, report)


class _Held:
    """A report that holds a trial segmentation's messages (its progress goes through), to
    be passed on only if that segmentation is the one kept."""

    def __init__(self, report):
        self.report, self.calls = report, []

    def __getattr__(self, name):
        if name in ("progress", "stage", "step", "section"):
            return getattr(self.report, name)

        def hold(*a, **k):
            self.calls.append((name, a, k))
        return hold

    def replay(self):
        for name, a, k in self.calls:
            getattr(self.report, name)(*a, **k)


def _thread(surf):
    return getattr(getattr(surf, "model", None), "kind", None) == "thread"


def _sure(seg, surf, faces):
    """Whether one of our seeds is sure: an exact surface that MMM's own accept test, at
    its tolerance, takes at least SURE_SHARE of the triangles of (our approximating
    surfaces, freeform patches, pipes and extrusions, never are)."""
    if isinstance(surf, OurSurface):
        return False
    faces = np.asarray(faces, dtype=np.int64)
    return float(seg.accept(surf, seg.sense_of(surf, faces), faces).mean()) >= SURE_SHARE


def _smooth_areas(seg):
    """Each triangle's smooth area: connected across edges whose triangles meet at less
    than PATCH_CREASE_DEG (as MMM's _find_sweeps cuts its patches)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    nf = len(seg.F)
    rows = np.repeat(np.arange(nf), 3)
    cols = seg.nb.reshape(-1)
    m = cols >= 0
    smooth = m.copy()
    smooth[m] = (seg.fn[rows[m]] * seg.fn[cols[m]]).sum(1) >= math.cos(math.radians(PATCH_CREASE_DEG))
    G = sparse.coo_matrix((np.ones(int(smooth.sum()), np.int8), (rows[smooth], cols[smooth])), shape=(nf, nf))
    return connected_components(G, directed=False)[1]


def _census(seg, comp, area):
    """(how many regions have triangles in the area, whether all of those are exact)."""
    n, exact = 0, True
    for reg in seg.regions:
        if len(reg.faces) and (comp[reg.faces] == area).any():
            n += 1
            exact = exact and reg.kind in EXACT_KINDS and not isinstance(reg.prim, OurSurface)
    return n, exact


def _winner(ours, theirs):
    """Which way an unsure area keeps, "ours" or "mmm", from each way's census (regions,
    all exact): the fewer regions, except that a way made only of exact surfaces (what a
    designer draws, and what edits well) beats one using freeform surfaces unless that one
    has at most FREEFORM_SAVES of its regions. A tie keeps ours, as before."""
    (na, ea), (nb, eb) = ours, theirs
    if ea and not eb:
        return "mmm" if nb <= FREEFORM_SAVES * na else "ours"
    if eb and not ea:
        return "ours" if na <= FREEFORM_SAVES * nb else "mmm"
    return "mmm" if nb < na else "ours"


def _segment(shell, seeds, tol, quality, report=NULL):
    """MMM's Segmenter for one shell, our regions in first (locked, grown into the
    neighbouring triangles that fit them by MMM's own test), then MMM's run over the rest."""
    seg = mcad.Segmenter(shell.V, shell.F, shell.topo, tol, report)
    seg.detail = quality
    # (no seeds on known axes here: our features hold those surfaces already, and the extra
    # hypotheses cost Solar Panel Bracket Large its CAD faces; see mmm/cad.py, b13 note)
    seg.known_axes = False
    for surf, faces in seeds:
        faces = np.asarray(faces, dtype=np.int64)
        faces = faces[seg.label[faces] < 0]
        if not len(faces):
            continue
        sense = seg.sense_of(surf, faces)
        seg.label[faces] = len(seg.regions)
        seg.regions.append(mcad.Region(surf, faces, sense))
    for rid, reg in enumerate(list(seg.regions)):
        grown, _ = seg.grow(reg.prim, reg.sense, reg.faces, refit=False)
        new = grown[seg.label[grown] < 0]
        if len(new):
            seg.label[new] = rid
            reg.faces = np.concatenate([reg.faces, new])
    seg.run()
    return seg


# ----------------------------------------------------------------------------
# The front end handed to MMM's pipeline
# ----------------------------------------------------------------------------


WRECKED = 0.02      # our repair moving the enclosed volume more than this share: try MMM's


def _signed_volume(pts, tris):
    a, b, c = (pts[tris[:, k]] for k in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


class _Route:
    """The method for each of our bodies, chosen before any feature is looked for (called
    by convert.prepare with the bodies as split): which to look for features in.

    --preset cad: every body (a mechanical part: our features, then MMM's segmentation of
    the rest). --preset smooth: none (an artistic model: smooth surfaces only), nor for
    precise, which MMM's own fitting does. Auto: MMM's own choice per body
    (pipeline.choose_preset_body: flat faces, one motion sweeping the curved surface, one
    surface a shell), as MMM alone makes it; a body routed to smooth surfaces is looked
    at only if they don't follow its mesh and CAD faces are tried (later). Before, any
    body we found anything in was forced to CAD: on organic_metaballs a patchwork of 342
    analytic pieces in 128 s, where MMM alone made 348 smooth surfaces in 17 s.
    pieces[k]: part k's pieces (_shells_of), made here and kept for bodies_from; want[k]:
    whether part k is looked at."""

    def __init__(self, opts, report):
        self.opts, self.report = opts, report
        self.pieces = self.want = None

    def __call__(self, pts, groups):
        self.pieces = [_shells_of(pts, g) for g in groups]
        self.want = self._choose(pts, groups)
        return self.want

    def _choose(self, pts, groups):
        preset = self.opts.preset
        if preset == "cad":
            self.report.log("CAD faces, as chosen (a mechanical part): stl2curves looks for features in every body")
            return [True] * len(groups)
        if preset != "auto":
            self.report.log(f"{mpipe.PRESETS.get(preset, {}).get('label', preset)} surfaces, as chosen: stl2curves' feature finding isn't used")
            return [False] * len(groups)
        biggest = max(_area(pts, g) for g in groups)
        out = []
        for gi, pieces in enumerate(self.pieces):
            made = _nest([p[0] for p in pieces], [[] for _ in pieces])
            kinds = []
            for b in made:
                if _debris(b.shells[0], biggest):
                    continue
                P = np.vstack([s.V for s in b.shells])
                tol = self.opts.tolerance or mpipe.auto_tolerance(float(np.linalg.norm(P.max(0) - P.min(0))), "cad",
                                                                  self.opts.quality)
                kinds.append(mpipe.choose_preset_body(b.shells, tol, self.opts.quality, NULL)[0])
            cad = "cad" in kinds or not kinds
            out.append(cad)
            which = f"Body {gi + 1}" if len(groups) > 1 else "The part"
            self.report.log(f"{which}: " + ("CAD faces (flat faces, a sweep, or one simple surface): stl2curves "
                                            "looks for features in it" if cad else
                                            "smooth surfaces (an organic shape: no flat faces to speak of): "
                                            "stl2curves' feature finding is skipped, and used only if CAD faces "
                                            "are tried after all"))
        return out


def _plate(path):
    """A 3MF project as (points, triangles, part of each triangle): every object its build
    prints, placed and scaled as printed (read3mf), each object's parts numbered apart so
    prepare() mends and builds each on its own (a part's pieces can overlap)."""
    objects = convert.read_3mf(path)
    if not objects:
        raise mio.MeshError("This 3MF project has nothing to print in it, so there's no shape to convert.")
    pts, tris, part, start, label = [], [], [], 0, 0
    for _, P, T, L in objects:
        pts.append(P)
        tris.append(T + start)
        part.append(np.asarray(L) + label)
        start += len(P)
        label += int(np.max(L)) + 1 if len(L) else 1
    return np.vstack(pts), np.vstack(tris), np.concatenate(part)


def _mended(path, opts, report, route=None):
    """Our prepared parts, on our mended mesh, or on MMM's where ours moved the enclosed
    volume by more than WRECKED and MMM's moved it less (a cap of thin walls: ours split
    4,825 of its 5,196 triangles away as slivers, -19% volume; MMM's kept it whole).
    Returns (parts, info, part, ours): ours, whether they are on our own mended mesh
    (saved as stages, which our own conversion alongside takes up again). Never for a 3MF
    plate: prepare() gets it as data, which saves no stages, and our own run would convert the
    whole plate again and write one STEP per object, under names it doesn't look for (found by
    Duvdivan's Claude in the review of #14)."""
    plate = str(path).lower().endswith(PLATE_FORMATS)
    source = _plate(path) if plate else path
    parts, info, part = convert.prepare(source, 0.01, blends=FRONT_BLENDS, analyze=route)
    pts0, tris0 = (source[0], source[1]) if plate else convert.load_stl(path)
    if not plate and any("three or more" in s for s in info.get("mesh_left") or []):
        # (edges of three or more triangles our repair couldn't remove: no solid closes there.
        # On a mesh finer than the weld grid they can be the grid's doing; read with only its
        # identical corners joined, if that is clean: see features.exact_weld)
        exact = F.exact_weld(path)
        if exact is not None:
            report.log("The mesh is finer than stl2curves' corner weld, which joined corners that should stay apart, "
                       "so it is read again with only its identical corners joined")
            parts, info2, part = convert.prepare(exact, 0.01, blends=FRONT_BLENDS, analyze=route)
            info2["triangles"] = info["triangles"]
            info = info2
            pts0, tris0 = exact
    raw = abs(_signed_volume(pts0, tris0))
    mine = abs(sum(_signed_volume(p[0].pts, p[0].body_tris) for p in parts))
    if not raw or abs(mine - raw) <= WRECKED * raw:
        return parts, info, part, not plate
    theirs_bodies, _ = mio.load_and_prepare(path, repair=opts.repair, remove_debris=opts.remove_debris, report=NULL,
                                            **({"layout": "plate"} if plate else {}))
    shells = [s for b in theirs_bodies for s in b]
    theirs = abs(sum(s.volume for s in shells))
    if abs(theirs - raw) >= abs(mine - raw):
        return parts, info, part, not plate
    report.log(f"stl2curves' repair moved the enclosed volume {100 * (mine - raw) / raw:+.1f}%, MMM's "
               f"{100 * (theirs - raw) / raw:+.1f}%: MMM's mended mesh is used, stl2curves' features found on it")
    offsets = np.cumsum([0] + [len(s.V) for s in shells])
    pts = np.vstack([s.V for s in shells])
    tris = np.vstack([s.F + o for s, o in zip(shells, offsets)])
    parts, info2, part = convert.prepare((pts, tris), 0.01, mend=False, blends=FRONT_BLENDS, analyze=route)
    info2["triangles"] = info["triangles"]
    info2["repairs"] = ["MMM's repair (stl2curves' moved the volume too far)"]
    return parts, info2, part, False


def front(path, opts, report):
    """stl2curves' first stages for MMM's pipeline.convert: (bodies, info, seeded,
    extra). seeded[k](tol) gives body k's ("cad", {id(shell): segmenter}); a body
    missing from it (an organic body with nothing of ours in it) gets MMM's own choice.
    extra[k]: more candidates for body k (our own sewn build).
    If our stages fail on the file (which passed the entrance, so MMM's reader takes it),
    MMM's own reading, mending and segmentation convert it instead, and the user is told.
    A mesh our repair leaves nothing of is refused (_NothingLeft)."""
    try:
        return _front(path, opts, report)
    except (MemoryError, _NothingLeft):
        raise
    except Exception as e:
        workers.stop()
        report.warn(f"stl2curves' feature finding failed on this file ({type(e).__name__}: {e}), so MMM's engine "
                    "converts it on its own: its own mesh repair and surface fitting.")
        bodies, info = mio.load_and_prepare(path, repair=opts.repair, remove_debris=opts.remove_debris, report=report,
                                            layout=opts.layout, merge=opts.merge_overlaps,
                                            touching=opts.touching_parts)
        return bodies, info, {}, {}


class _NothingLeft(mio.MeshError):
    """Our repair left no body: every piece of the mesh was loose debris. 3,000 triangles
    lying apart (a triangle soup) crashed here; MMM alone takes each one for a body of
    its own and was still at it after 5 minutes, to save 3,000 loose flat faces."""


def _front(path, opts, report):
    if str(path).lower().endswith(OTHER_FORMATS):
        # (OBJ, PLY and OFF: MMM's own reader and segmentation, for now)
        bodies, info = mio.load_and_prepare(path, repair=opts.repair, remove_debris=opts.remove_debris,
                                            report=report, layout=opts.layout, merge=opts.merge_overlaps,
                                            touching=opts.touching_parts)
        return bodies, info, {}, {}
    report.stage("Reading and mending the mesh (stl2curves)", 0.0, 0.3)
    route = _Route(opts, report)
    parts, info, part, parts_on_file = _mended(path, opts, report, route)
    groups = [p[0].__dict__.get("body_tris") for p in parts]
    if any(g is None for g in groups):
        raise RuntimeError("prepare() didn't keep each body's triangles")
    pts = parts[0][0].pts
    report.stage("Handing the features over (stl2curves -> MMM)", 0.3, 0.35)
    _fit_surfaces(parts)
    ours = bodies_from(parts, groups, pts, route.pieces)
    bodies = [b.shells for b in ours]
    if not bodies:
        raise _NothingLeft(f"Nothing is left of this mesh once its loose bits are removed: its {info['triangles']:,} "
                           "triangles don't join up into a surface, so there's no shape to convert.")
    minfo = mio.MeshInfo()
    allP = np.vstack([s.V for b in bodies for s in b])
    minfo.diag = float(np.linalg.norm(allP.max(0) - allP.min(0)))
    minfo.bbox = (allP.min(0), allP.max(0))
    minfo.triangles = minfo.input_triangles = info["triangles"]
    minfo.vertices = len(allP)
    minfo.shells = sum(len(b) for b in bodies)
    minfo.open_shells = sum(not s.closed for b in bodies for s in b)
    minfo.bodies = len(bodies)
    minfo.body_names = [None] * len(bodies)
    minfo.part_names = ["part"]
    minfo.part_triangles = [info["triangles"]]
    minfo.body_part = [0] * len(bodies)
    minfo.mesh_check = {"status": "ok", "issues": [], "repairs": info.get("repairs", [])}
    n_open = sum(1 for b in bodies if not all(s.closed for s in b))
    if n_open:
        # (as MMM's mesh check says it: a sheet or a single triangle came back "valid" with
        # no word that the file holds a surface, not a solid)
        from mmm.meshcheck import _issue, _plural
        minfo.mesh_check["issues"].append(_issue(
            "open_surface", n_open, "not_fixed", f"{_plural(n_open, 'piece')} can't be closed into a solid (a "
            "surface with no thickness, or holes too big to fill)", "Converted as surfaces instead of a solid."))
        minfo.mesh_check["status"] = "problems"
        report.warn(f"{_plural(n_open, 'piece')} of this mesh can't be closed into a solid, so "
                    + ("it is" if n_open == 1 else "they are") + " converted as surfaces instead of a solid.")
    # (our repair can close a single triangle into two faces back to back: "a solid" of no
    # volume, which came back valid with no word)
    n_flat = sum(1 for b in bodies if all(s.closed for s in b)
                 and abs(sum(s.volume for s in b)) <= FLAT * minfo.diag ** 3)
    if n_flat:
        from mmm.meshcheck import _issue, _plural
        minfo.mesh_check["issues"].append(_issue(
            "no_volume", n_flat, "not_fixed", f"{_plural(n_flat, 'piece')} enclose{'s' if n_flat == 1 else ''} "
            "no volume (flat, or paper-thin)", "Converted, but not as a real solid."))
        minfo.mesh_check["status"] = "problems"
        report.warn(f"{_plural(n_flat, 'piece')} of this mesh enclose{'s' if n_flat == 1 else ''} no volume (flat, "
                    "or paper-thin), so the result is not a real solid.")
    report.log("stl2curves mended the mesh: " + ("; ".join(info["repairs"]) if info.get("repairs") else "nothing to mend"))

    def seeded(k):
        body = ours[k]
        if not any(body.seeds):
            return None

        def make(tol):
            segs = {}
            for shell, seeds in zip(body.shells, body.seeds):
                segs[id(shell)] = seeded_segmenter(shell, seeds, tol, opts.quality, report)
            return "cad", segs
        return make

    # Our own conversion, faces cut and sewn, runs alongside in a process of its own from
    # here (see _OurRun); our worker processes here aren't needed any more, it has its own.
    # They are stopped, not kept for a next part as stl2curves keeps them: each holds a copy
    # of the mesh, and three merged conversions at once, each with this pool and our own
    # run's, took 38 GB in 89 processes on medium meshes
    run = _OurRun(path) if parts_on_file and any(p[1] for p in parts) and getattr(opts, "rivals", RIVALS) else None
    workers.stop()

    def sewn(k):
        """Body k from our own conversion, as one more candidate for MMM's judge: tried
        first when the merged faces don't pass. A part of several bodies takes, for each,
        the solid matching its outer shell."""
        body = ours[k]

        def make():
            shape = run.shape(give_up=opts.enough)
            if shape is None:
                return None
            return shape if len(ours) == 1 else _nearest_solid(shape, body.shells[0])
        return [("stl2curves", "Building stl2curves' own (sewn) faces", make)]

    analysed = {}

    def later(k):
        """Body k's segmentation seeded with our features, looked for only now: for a body
        routed to smooth surfaces when CAD faces are tried after all (pipeline marks it by
        .later). ("cad", None) if we find nothing in it: MMM's own segmentation."""
        body = ours[k]
        gi = body.part

        def make(tol):
            if gi not in analysed:
                report.log("stl2curves looks for features in this body now, as CAD faces are tried for it")
                g = groups[gi]
                if len(g) >= convert.WORKERS_FROM:
                    workers.start(len(g))
                try:
                    mesh, features, t = convert._analyze_body(pts, g, 0, 1)
                    convert.snap(mesh, features, None)
                    if FRONT_BLENDS:
                        features = convert.add_blends(mesh, features)
                finally:
                    workers.stop()
                analysed[gi] = part_seeds(mesh, features, pts, g, route.pieces[gi])
            seeds = analysed[gi]
            body.seeds = [seeds[j] for j in body.pieces]
            if not any(body.seeds):
                return "cad", None
            return make_segs(body, tol)
        make.later = True
        return make

    def make_segs(body, tol):
        segs = {}
        for shell, seeds in zip(body.shells, body.seeds):
            segs[id(shell)] = seeded_segmenter(shell, seeds, tol, opts.quality, report)
        return "cad", segs

    made = {k: seeded(k) if route.want[b.part] else (later(k) if opts.preset == "auto" else None)
            for k, b in enumerate(ours)}
    extra = {k: sewn(k) for k in range(len(bodies)) if run is not None and parts[ours[k].part][1]}
    return bodies, minfo, {k: m for k, m in made.items() if m is not None}, extra


_running = []   # our conversions started alongside (stopped when the file is done)


class _OurRun:
    """stl2curves' own conversion of the file (the MIT pipeline: faces cut and sewn,
    culprits dropped) in a process of its own, started once our stages are saved: it
    takes them up again (MM_S2C_CACHE) and goes straight to building, on the cores
    MMM's single-threaded stages leave free, so its result is usually there by the time
    MMM's judge asks for it (Rak N-USB waited 110 s for it run afterwards). Also a rival
    to the merged result for the whole file (rivals.py)."""
    name = "stl2curves"

    def __init__(self, path):
        import subprocess
        import tempfile
        self.dir = tempfile.mkdtemp(prefix="mms2c_ours_")
        self.out = os.path.join(self.dir, os.path.splitext(os.path.basename(str(path)))[0] + ".step")
        env = dict(os.environ, MM_S2C_CACHE=stages.FOLDER)
        self.proc = rivals.popen(["-m", "mms2c", str(path), "--out", self.dir], env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._shape = False
        _running.append(self)

    def shape(self, give_up=None):
        """The finished conversion's shape, waiting for it (None if it failed, or if
        give_up() turns true first: the run itself goes on, as a rival)."""
        if self._shape is False:
            while self.proc.poll() is None:
                if give_up is not None and give_up():
                    return None
                time.sleep(0.5)
            self._shape = None
            if self.proc.returncode == 0 and os.path.exists(self.out):
                from mmm import stepio
                try:
                    self._shape = stepio.read_step(self.out)
                except Exception:
                    self._shape = None
        return self._shape

    def stop(self):
        import shutil
        rivals.kill(self.proc)
        shutil.rmtree(self.dir, ignore_errors=True)


BOX_MATCH = 0.05    # a solid stands for a body when its box is within this share of the body's size


def _nearest_solid(shape, shell):
    """The solid of shape whose box matches the shell's (None if none does: our build
    can fuse a small body into the one it touches, and the whole part judged against
    the small body's mesh took 14.7 GB and ran for over half an hour)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    lo, hi = shell.V.min(0), shell.V.max(0)
    best, gap = None, np.inf
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    while ex.More():
        s = ex.Current()
        ex.Next()
        box = Bnd_Box()
        BRepBndLib.Add_s(s, box)
        d = np.abs(np.array(box.CornerMin().Coord()) - lo).sum() + np.abs(np.array(box.CornerMax().Coord()) - hi).sum()
        if d < gap:
            best, gap = s, d
    return best if gap <= BOX_MATCH * float(np.linalg.norm(hi - lo)) * 6 else None


def _rivals_fit(base):
    """Whether the rivals can make what these settings ask for: they convert in
    millimetres at the engine's own tolerance with no face limit, so a file asked for in
    inches, at a tolerance or with a face limit is made by the merged pipeline alone."""
    return base is None or (base.unit == "mm" and not base.tolerance and not base.max_faces)


def _entrance(path, out):
    """The checks every file passes before any work: (whether it is read as an STL, its
    triangle count, its diagonal; 0 and None for other formats). Raises what mmm/cli.py's failure() turns into
    a message for the user: the paths (pipeline.check_paths: an output naming the input
    overwrote the user's mesh), then the mesh, read by MMM's reader and given its
    refusals (meshio.check_triangles: our own reader crashed on empty and malformed
    files, and a sphere 1e7 mm from the origin, all corners joined in MMM's weld, got
    past to a quad remesh that took 14 GB). Other formats are checked by MMM's reader
    in its own pipeline."""
    mpipe.check_paths(path, out)
    if str(path).lower().endswith(OTHER_FORMATS):
        return False, 0, None
    if str(path).lower().endswith(PLATE_FORMATS):
        try:
            pts, tris, _ = _plate(path)
        except (mio.MeshError, OSError, MemoryError):
            raise
        except Exception as e:
            raise mio.MeshError(f"This 3MF project seems to be damaged, so it couldn't be read ({type(e).__name__}). "
                                "Try saving it again.") from e
        mio.check_triangles(pts[tris].astype(np.float32))
        return True, len(tris), float(np.linalg.norm(pts.max(0) - pts.min(0)))
    try:
        tris = mio.read_stl(str(path))
    except (mio.MeshError, OSError, MemoryError):
        raise
    except Exception as e:
        raise mio.MeshError(f"This file seems to be damaged, so it couldn't be read ({type(e).__name__}). "
                            "Try exporting it again.") from e
    mio.check_triangles(tris)
    pts = tris.reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    return True, len(tris), float(np.linalg.norm(pts.max(0) - pts.min(0)))


class _PartReporter(Reporter):
    """The reporter for one part of a 3MF project split into parts (_convert_parts): its
    stages and progress go to the parent's, squeezed into the part's share [lo, hi] and
    named after the part; its result is kept, not sent (the parent sends a "part" event)."""

    def __init__(self, parent, lo, hi, name):
        super().__init__(parent.json_mode, parent.quiet, parent.stream)
        self.t0, self.parent, self.lo, self.hi, self.name, self.res = parent.t0, parent, lo, hi, name, None

    def _emit(self, obj):
        if obj.get("event") == "result":
            self.res = obj
            return
        if "progress" in obj:
            obj["progress"] = self.lo + (self.hi - self.lo) * obj["progress"]
        if obj.get("event") == "stage":
            obj["stage"] = f"{self.name}: {obj['stage']}"
        Reporter._emit(self.parent, obj)


def _write_stl(path, V, F):
    """A binary STL of (vertices, triangles)."""
    T = V[F].astype(np.float32)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, T
    with open(path, "wb") as fh:
        fh.write(b"Mesh Mix-S2C part".ljust(80, b" "))
        fh.write(np.uint32(len(F)).tobytes())
        fh.write(rec.tobytes())


def _convert_parts(path, out, quality_, report, preset, base):
    """A 3MF project split into parts: every part the app lists (MMM's reader, with the
    app's layout, overlap and touching-parts settings, so the parts and their numbers are the
    ones the app shows and names files by) converted on its own through the merged engine,
    one STEP each, as MMM's pipeline does for a split project (pipeline._convert_split).
    Each part goes in as MMM's mended mesh of it. None when the project has one part (the
    whole-plate path converts it)."""
    report = report or NULL
    t0 = time.time()
    report.stage("Reading the 3MF project", 0.0, 0.05)
    bodies, info = mio.load_and_prepare(path, repair=base.repair, remove_debris=base.remove_debris, report=NULL,
                                        layout=base.layout, merge=base.merge_overlaps, touching=base.touching_parts)
    if info.parts < 2:
        return None
    report.log(f"Split into {info.parts} parts: each is converted on its own with the merged engine")
    tmp = tempfile.mkdtemp(prefix="mms2c_parts_")
    used = set(o for o in base.part_outputs if o)
    results = []
    try:
        for p, name in enumerate(info.part_names):
            shells = [sh for b, bp in zip(bodies, info.body_part) if bp == p for sh in b]
            if not shells:
                continue
            offsets = np.cumsum([0] + [len(sh.V) for sh in shells])
            V = np.vstack([sh.V for sh in shells])
            tris = np.vstack([sh.F + o for sh, o in zip(shells, offsets)])
            src = os.path.join(tmp, f"part{p + 1}.stl")
            _write_stl(src, V, tris)
            dest = (base.part_outputs[p] if p < len(base.part_outputs) and base.part_outputs[p]
                    else mpipe._part_output_path(str(out), name, used))
            sub = mpipe.Options(**base.__dict__)
            sub.split, sub.part_outputs, sub.part_previews = False, [], []
            sub.preview = base.part_previews[p] if p < len(base.part_previews) and base.part_previews[p] else None
            sub.product_name = name
            lo, hi = 0.05 + 0.95 * p / info.parts, 0.05 + 0.95 * (p + 1) / info.parts
            rep = _PartReporter(report, lo, hi, name)
            t_part = time.time()
            try:
                got = convert_file(src, dest, quality_, rep, preset=preset, base=sub) or rep.res or {}
            except Exception as e:
                report.warn(f"The part “{name}” couldn't be converted and was left out ({type(e).__name__}: {e}).")
                continue
            r = {k: got.get(k) for k in ("preset", "presets", "valid", "solids", "shells", "faces", "edges", "volume",
                                         "mesh_volume", "file_size", "fidelity", "checked", "check_ok", "checks",
                                         "repaired")}
            r.update({"index": len(results), "part": p, "name": name, "output": dest, "preview": sub.preview,
                      "seconds": round(time.time() - t_part, 2),
                      "triangles": info.part_triangles[p] if p < len(info.part_triangles) else len(tris)})
            r["presets"] = r["presets"] or ([r["preset"]] if r["preset"] else [])
            report.part(r)
            results.append(r)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if not results:
        raise mpipe.ConversionError("None of the parts could be converted.")
    presets = [r["preset"] for r in results]
    res = {
        "output": results[0]["output"],
        "outputs": [r["output"] for r in results],
        "split": True,
        "parts": results,
        "preset": presets[0] if all(x == presets[0] for x in presets) else "mixed",
        "presets": sorted({x for r in results for x in r["presets"]}),
        "valid": all(r["valid"] is not False for r in results) if base.validate else None,
        "seconds": round(time.time() - t0, 2),
        "input": info.as_dict(),
        "mesh_check": info.mesh_check,
        # (as pipeline._convert_split sums them up: the apps show these for the whole project)
        "checked": all(r.get("checked") for r in results),
        "check_ok": all(r.get("check_ok") for r in results),
        "checks": [{**i, "message": f"{r['name']}: {i.get('message', '')}"} for r in results for i in r.get("checks") or []],
        "repaired": mpipe.repair_steps(results),
        "fidelity": mpipe.fidelity_summary([r["fidelity"] for r in results if r.get("fidelity")]),
    }
    for k in ("solids", "shells", "faces", "edges", "volume", "mesh_volume", "file_size"):
        res[k] = sum(r[k] or 0 for r in results)
    report.result(res)
    return res


def _convert_other(path, out, quality_, report, preset, base):
    """An OBJ, PLY or OFF file through the merged engine (Tuna's choice, 2026-10-31: the new
    engine for everything): its triangles, as read by MMM's reader, are saved as an STL
    and converted as one."""
    tmp = tempfile.mkdtemp(prefix="mms2c_mesh_")
    try:
        T = mio.read_any(str(path))
        src = os.path.join(tmp, os.path.splitext(os.path.basename(str(path)))[0] + ".stl")
        _write_stl(src, T.reshape(-1, 3), np.arange(3 * len(T)).reshape(-1, 3))
        sub = mpipe.Options(**(base.__dict__ if base is not None else {}))
        sub.product_name = sub.product_name or os.path.splitext(os.path.basename(str(path)))[0]
        return convert_file(src, out, quality_, report, preset=preset, base=sub)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def convert_file(path, out, quality_=None, report=None, preset="auto", base=None):
    """Convert one mesh file with the merged engine into the STEP file out: MMM's result
    dict (faces, validity, how closely it follows the mesh...). preset: "auto" (the
    method chosen per body, see _Route), "cad" (a mechanical part: our features and CAD
    faces for every body), "smooth" (an artistic model: smooth surfaces, no feature
    finding), or MMM's "precise". base: the settings the engine's command line was given
    (mmm.pipeline.Options: unit, tolerance, repair, preview...), which the merged
    pipeline keeps."""
    if base is not None and base.split and str(path).lower().endswith(PLATE_FORMATS):
        res = _convert_parts(path, out, quality_, report, preset, base)
        if res is not None:
            return res
    if str(path).lower().endswith(OTHER_FORMATS):
        return _convert_other(path, out, quality_, report, preset, base)

    report = report or Reporter()
    t0 = time.time()
    if quality_ is not None:
        mpipe.Options(quality=quality_)     # (refuses NaN and out of range: OptionError)
    stl, n, diag = _entrance(path, out)
    if stl and not FRONT_SIZES[0] <= diag <= FRONT_SIZES[1]:
        # (MMM's mesh check then gives its own note on the unit)
        report.warn(f"This model is {diag:.3g} units across, outside the sizes stl2curves' feature finding "
                    "works at, so Manifold Mesh Mix's engine converts it on its own")
        stl = False
    elif stl and not UNIT_NOTE[0] <= diag <= UNIT_NOTE[1]:
        report.warn(f"The model is {diag:,.3g} units across. If it was made in "
                    + ("inches or meters" if diag < UNIT_NOTE[0] else "a smaller unit than millimeters")
                    + ", change Units in the settings.")
    q = quality_ if quality_ is not None else quality(n)
    use = stl and RIVALS and _rivals_fit(base)
    # MMM's engine alone from the start, and our own conversion once our first stages are
    # saved (see _OurRun): rivals to the merged result (see rivals.py). (3MF projects go
    # to MMM's own reader unseeded, so MMM alone would only repeat them.)
    # (not for a 3MF plate: MMM alone would read it in its parts' own frames)
    if use and not str(path).lower().endswith(PLATE_FORMATS):
        _running.append(rivals.MMMAlone(path, q))
    judges = rivals.Judges(path, _running)
    opts = mpipe.Options(**(base.__dict__ if base is not None else {}))
    opts.preset, opts.quality = preset, q
    opts.judge_floor, opts.fewest_faces, opts.enough = rivals.JUDGE_FLOOR, True, judges.passing
    opts.smooth_elsewhere = opts.rivals = use
    own_cache = not stages.FOLDER
    if own_cache:
        stages.FOLDER = tempfile.mkdtemp(prefix="mms2c_stages_")
    try:
        try:
            try:
                with judges.run_lock:
                    judges.running = True
                res = mpipe.convert(str(path), str(out), opts, report, front=front if stl else None)
            finally:
                with judges.run_lock:
                    judges.running = False
        except rivals.Overtime:
            # (a rival passed and the merged pipeline ran past its deadline: rivals.DEADLINE)
            report.log("The merged pipeline was stopped: a conversion alongside already follows the mesh, "
                       "and it had run past its deadline")
            res = None
            workers.stop()      # (its worker jobs may still be running)
        if _running:
            res = rivals.settle(out, res, judges, time.time() - t0, report)
            res["seconds"] = round(time.time() - t0, 2)
            if res.get("kept", "merged") != "merged" and opts.preview:
                _preview_again(out, opts.preview, report)
            report.result(res)
        return res
    finally:
        judges.close()
        while _running:
            _running.pop().stop()
        workers.finish()
        if own_cache:
            shutil.rmtree(stages.FOLDER, ignore_errors=True)
            stages.FOLDER = None


def _preview_again(step, preview, report):
    """The app's preview of the file a rival made (the merged pipeline's preview shows its
    own result)."""
    from mmm import stepio
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    try:
        shape = stepio.read_step(step)
        box = Bnd_Box()
        BRepBndLib.Add_s(shape, box)
        diag = float(np.linalg.norm(np.subtract(box.CornerMax().Coord(), box.CornerMin().Coord())))
        stepio.export_preview(shape, preview, diag)
    except Exception as e:
        report.warn(f"The 3D preview couldn't be made ({e}); the STEP file itself is fine.")


_job = None     # the job object of _end_with_me (its handle must live as long as the process)


def _end_with_me():
    """Put this process in a Windows job object that ends every process in it when its
    last handle closes, which is when this process ends, however it ends: ended from
    outside (the task manager, a test's time limit), the merged engine left its rivals,
    our own run and their worker pools running (25 processes after a 10 GB conversion).
    Children join the job as they start. Returns the handle, kept for the process's life
    (None if it couldn't be made, or off Windows). From Mesh-Mix-S2C's engine/mmm_engine.py
    (2026-10-26), where every engine process starts this way."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", ctypes.c_ulonglong * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateJobObjectW.restype = wintypes.HANDLE
        k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        job = k32.CreateJobObjectW(None, None)
        if not job:
            return None
        info = Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))  # extended limits
        if not ok or not k32.AssignProcessToJobObject(job, k32.GetCurrentProcess()):
            return None
        return job
    except Exception:
        return None


def main(argv=None):
    import argparse
    import json
    import os
    ap = argparse.ArgumentParser(prog="stl2curves (merged)", description=__doc__.strip().splitlines()[0])
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--quality", type=float, help="MMM's quality, 0-1 (default: by triangle count)")
    ap.add_argument("--json", action="store_true", help="progress as JSON lines on stdout")
    ap.add_argument("--preset", choices=("auto", "cad", "smooth", "precise"), default="auto",
                    help="cad: a mechanical part; smooth: an artistic model; auto (default): chosen for each "
                         "body by its shape")
    a = ap.parse_args(argv)
    out = a.output or os.path.splitext(a.input)[0] + ".step"
    report = Reporter(json_mode=a.json)
    global _job
    if argv is None:            # (the command line, not a script or a test calling main)
        _job = _end_with_me()
    try:
        res = convert_file(a.input, out, a.quality, report, a.preset)
    except Exception as e:
        # (the same messages as MMM's command line, and a JSON error event with --json)
        from mmm.cli import failure
        return failure(e, a.input, report)
    if not a.json:
        print(json.dumps({k: v for k, v in res.items() if k != "input"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
