"""CAD preset: recognise analytic surfaces and build an editable B-rep."""
# Part of Manifold Mesh Mix (c) 2026 Tuna-The-First and Claude, AGPL-3.0, commit 4cf72ea.
# Modified 2026-10-25 for the stl2curves merged engine (mms2c/merged.py): surfaces
# stl2curves found come in "locked" (_locked): borders move, the surface is never refitted,
# simplified, swallowed by a sweep or a fillet chain, and survives merges; faces still
# broken after build_cad_shell's rounds become flat faces of their own triangles.
# Modified 2026-10-26: after its rounds build_cad_shell checks the faces once more written
# as one shell, as the file holds them, and faces broken that way become flat faces of their
# own triangles (the rounds' check writes a loose compound, which read back clean while the
# saved file broke two faces: Rak Uni-USB); a triangle whose direction isn't trusted
# (Segmenter.unsure) must still face within the tilt noise can give it (Segmenter.cos_ok).
# Modified 2026-10-28: convert_body_cad keeps the healed shape only when it is better
# (pipeline.fix_acceptable: valid, volume of the mesh's sign and size), and turns a body
# that comes out inside out the right way round (pipeline.orient_like_mesh).
# Modified 2026-10-28 (b2): a freeform cell _make_freeform fits over weak flat faces that
# can't follow its own triangles (further off than the tolerance, or the mesh's own
# coarseness where that is larger) goes back to those flat faces, which are exact; each
# cell keeps the flat faces it was made from (Region.parts). Toolbox bottom: a wall and its
# 30-degree chamfer became one B-spline bulging 0.42 mm.
# Modified 2026-10-28 (group C): a cone face round its apex is built as a cap with a
# degenerate edge at the apex, as a sphere's cap round its pole was (plan_seams,
# _pole_cap_wire, _generator).
# Modified 2026-10-28 (group C): a cylinder, cone or torus region winding more than one
# turn round its axis is cut into one face a turn at a meridian (split_turns), with its
# seam and the corners of the cut on that meridian (Region.cut, snap_turn_cuts).
# Modified 2026-10-28 (group C): a torus face round its tube but only part way round its
# axis (a bent rod or bore) gets a seam along one of the torus's parallels
# (_plan_tube_seam, _tube_seam_wire), as a face round the axis gets one along a meridian.
# Modified 2026-10-28 (group E): _straighten_borders settles a tie between equally short
# borders by the triangles' shape (_tied_side: normal agreement, curved over flat, larger
# region, position), not by which region came first, which made the choice depend on
# region order. (The mirrored side brace's -1.15% volume has another cause, still open: the
# cut weighs border length only, not how well each surface fits the rows it moves.)
# Modified 2026-10-28 (b13): _process_seed also fits each seed on the known axes near it
# (_axis_hypotheses): the axes of stl2curves' locked surfaces and threads, and of the
# segmenter's own cylinders, cones and tori once they reach AXIS_AROUND_DEG round; only
# the profile is fitted, to the seed's own corners, so a narrow band (a chamfer, a
# countersink) no longer loses to a plane because its neighbourhood took in the faces
# beside it (Rak Front Lid through MMM alone: 204 -> 102 faces; worm_608bearing 358 -> 32).
# Off where the merged engine seeds the segmenter with stl2curves' features (Segmenter.
# known_axes): they hold those axes' surfaces already, and there it cost Solar Panel Bracket
# Large its CAD faces (one smooth area 22 -> 24 regions tipped compete the wrong way).
# Modified 2026-10-29 (slivers): a triangle with no area to speak of goes to the region across
# its longest edge before each build (Segmenter._flat_slivers_to_long_edge): held by a short
# edge it made a zero-width spike in its face's outline, which OpenCASCADE finds crossing.
# Modified 2026-10-29 (brace-fit): _straighten_borders' minimum cut also weighs how well each
# surface fits the triangles it may move (FIT_WEIGHT), not border length only: of equally
# short borders, the one keeping triangles on the surface they fit better wins.
# Modified 2026-10-31 (Tuna's side): a flat face made from a region's own triangles that is
# still broken is built once more from truly coplanar triangles only (FACET_FLAT): grouped
# within FACET_GROUP of its plane, its corners stood as far off that plane as its edges'
# tolerance, and the check failed on the tie (Solar Arm Mount was saved faceted for two).
# Modified 2026-10-31 (Tuna's side): _heal_faces no longer keeps a repair that turns a
# neighbouring face inside out (HEAL_GROWTH): the edge it hands on can leave a periodic
# neighbour with everything but its patch (a whole torus round Bravo's motor shaft joint).
import numpy as np

from OCP.BRep import BRep_Tool
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepLib import BRepLib
from OCP.Geom import (
    Geom_ConicalSurface,
    Geom_CylindricalSurface,
    Geom_SphericalSurface,
    Geom_ToroidalSurface,
)
from OCP.GeomAbs import GeomAbs_C2
from OCP.GeomAPI import GeomAPI_PointsToBSpline
from OCP.ShapeFix import ShapeFix_Edge, ShapeFix_Wire
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp, TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Edge, TopoDS_Face, TopoDS_Vertex, TopoDS_Wire
from OCP.collections import IndexedMap_TopoDS_Shape_TopTools_ShapeMapHasher as _ShapeMap
from OCP.Geom2d import Geom2d_Line
from OCP.gp import gp_Ax2, gp_Ax3, gp_Circ, gp_Dir, gp_Dir2d, gp_Pnt2d
from OCP.collections import Array1_gp_Pnt

from .brep import MIN_TOL, BRepAssembler, _pnt, make_compound
from .fitting import CURVED, Cone, Cylinder, Plane, Sphere, Torus, _OnAxis, on_axis, perp_basis, unit
from .meshio import face_normals_areas
from .planar import planar_regions
from .report import NULL
from .topology import RegionGraph

# Surfaces on known axes (Segmenter._axis_hypotheses)
AXIS_AROUND_DEG = 90.0  # degrees: a fitted cylinder, cone or torus lends its axis once it reaches this far round
AXIS_REACH = 0.05       # share of the model's diagonal, past twice the surface's radius, an axis reaches
AXIS_NEAREST = 4        # the nearest so many axes in reach are tried for each seed
AXIS_TIE = 0.9          # a surface on a known axis wins over a free fit covering up to 1 / this as much
AXIS_SAME_DEG = 0.5     # degrees (and 4 tolerances apart at most): two axes are one

# Flat faces made from a region's own triangles (_flat_facets)
FACET_GROUP = 0.1       # tolerances: triangles share one flat face when this close to its plane ...
FACET_FLAT = 1e-3       # ... and only this close when that face broke (a second try, see build_cad_shell)

# Repairing the last broken faces (_heal_faces)
HEAL_GROWTH = 3.0       # a face this many times the area of its own triangles after a neighbour's repair ...
HEAL_GROWTH_MM2 = 1.0   # ... and this much more (mm^2) was turned inside out by it
HEAL_TRIES = 3          # times the repair is run again without the broken faces next to such a face


# ----------------------------------------------------------------------------
# Segmentation
# ----------------------------------------------------------------------------


def _locked(prim):
    """A surface stl2curves found (merged engine): kept exactly as it came."""
    return getattr(prim, "locked", False)


class Region:
    __slots__ = ("prim", "faces", "sense", "weak", "facet", "parts", "cut")

    def __init__(self, prim, faces, sense, weak=False):
        self.prim = prim
        self.faces = faces
        self.sense = sense
        self.weak = weak
        self.facet = False
        # a freeform cell made from weak flat faces: those regions (see _make_freeform)
        self.parts = None
        # one turn of a surface winding several (split_turns): the direction from
        # its axis of the meridian it was cut at, where its seam goes
        self.cut = None

    @property
    def kind(self):
        return self.prim.kind


class Segmenter:
    def __init__(self, V, F, topo, tol, report=NULL, normal_deg=25.0, smooth_deg=50.0, extent_deg=50.0):
        self.V = V
        self.F = F
        self.topo = topo
        self.tol = tol
        self.report = report
        self.fn, self.fa = face_normals_areas(V, F)
        nf = len(F)
        t = topo.twin.reshape(nf, 3)
        self.nb = np.where(t >= 0, t // 3, -1)
        self.ext = float(np.linalg.norm(V.max(0) - V.min(0)))
        self.cos_n = np.cos(np.radians(normal_deg))
        self.cos_smooth = np.cos(np.radians(smooth_deg))
        self.cos_extent = np.cos(np.radians(extent_deg))
        self.tiny = self.fa < 1e-10 * self.ext ** 2
        # needle triangles: thinner than the tolerance, so their normal says
        # little about the surface they belong to
        T = V[F]
        lmax = np.linalg.norm(T - np.roll(T, -1, axis=1), axis=2).max(1)
        self.needle = 2.0 * self.fa < 0.5 * tol * np.maximum(lmax, 1e-30)
        self.deflection, defl_hi = self._estimate_deflection()
        # surfaces must also follow the mesh between its vertices, within the
        # mesh's own tessellation coarseness
        self.tol_mid = max(3.0 * tol, 5.0 * self.deflection, 2.0 * defl_hi)
        # A scan's noise tilts thin triangles: one 0.03 mm high on a mesh with
        # 0.01 mm of noise points anywhere within about 40 degrees, so judged
        # by its direction it fits no surface, and a scanned flat face fell
        # apart into thousands of pieces (a plate with five holes: 38,830
        # faces). Measured, noise tilts a triangle of height h by about
        # atan(3.4 noise / h); where that exceeds the angle surfaces are
        # matched by, only the triangle's position counts. On such a mesh
        # flat faces are also fitted again as they grow (a small seed's own
        # plane is tilted by the noise). Clean meshes have no noise: none of
        # this applies to them.
        from .fidelity import mesh_noise

        self.noise = mesh_noise(V, F)
        h = 2.0 * self.fa / np.maximum(lmax, 1e-30)
        noisy = h < 3.4 * self.noise / np.tan(np.radians(normal_deg))
        self.scan = self.noise > 0.5 * tol or self.fa[noisy].sum() > 0.005 * max(self.fa.sum(), 1e-30)
        # triangles whose direction says nothing: tiny ones, and on a scan
        # the ones thin enough for the noise to tilt
        self.unsure = self.tiny | noisy if self.scan else self.tiny
        # the direction each triangle must agree with a surface's within: a thin
        # one on a scan can be tilted by noise as measured above, so it may
        # disagree by that much more, but not by more (tiny ones by anything).
        # Taken as no limit at all, the thin triangles of a blind hole's floor,
        # every corner on the hole's rim, joined its wall on a CAD mesh taken for
        # a scan by its slivers (Rak Uni-USB: 34 of them, at 90 degrees to it),
        # and the wall's border cut 51 degrees of the rim straight across, 0.17
        # mm inside it
        tilt = np.arctan(3.4 * self.noise / np.maximum(h, 1e-30))
        self.cos_ok = np.where(self.tiny, -1.0, np.where(
            self.unsure, np.cos(np.minimum(np.pi, np.radians(normal_deg) + tilt)), self.cos_n))
        self.refit_planes = self.scan
        self.label = -np.ones(nf, dtype=np.int64)
        self._stamp = np.zeros(nf, dtype=np.int64)
        self._stamp_id = 0
        self.regions = []
        self._axes, self._axes_n = [], -1

    # --- helpers --------------------------------------------------------
    def _estimate_deflection(self):
        """The chordal deflection the mesh was tessellated with (see
        fidelity.chord_deflection): (typical, high)."""
        from .fidelity import chord_deflection

        twin = self.topo.twin
        h = np.nonzero(twin >= 0)[0]
        t = twin[h]
        m = h < t
        h, t = h[m], t[m]
        L = np.linalg.norm(self.V[self.topo.he_to[h]] - self.V[self.topo.he_from[h]], axis=1)
        return chord_deflection(self.fn, self.fa, h // 3, t // 3, L, tiny=self.tiny)

    def _new_stamp(self):
        self._stamp_id += 1
        return self._stamp_id

    def face_error(self, prim, sense, faces):
        """(max vertex distance, cosine to primitive normal) for faces.

        A triangle whose corners lie on the surface can still be a poor match
        if it spans a large piece of it (e.g. a chamfer whose corners happen
        to lie on a sphere). Such triangles get cosine -1 (rejected): the
        surface normals at its corners must agree within ``extent_deg``.
        """
        P = self.V[self.F[faces]]
        flat = P.reshape(-1, 3)
        d = prim.distance(flat).reshape(-1, 3).max(1)
        nrm = prim.normal(P.mean(1)) * sense
        cs = (self.fn[faces] * nrm).sum(1)
        if prim.kind != "plane":
            M = np.concatenate([(P[:, 0] + P[:, 1]) / 2, (P[:, 1] + P[:, 2]) / 2, (P[:, 2] + P[:, 0]) / 2])
            dm = prim.distance(M).reshape(3, -1).max(0)
            # a mid-edge miss beyond the tessellation coarseness disqualifies
            d = np.where(dm <= self.tol_mid, d, np.inf)
        if prim.kind != "plane":
            nv = prim.normal(flat).reshape(-1, 3, 3)
            c01 = (nv[:, 0] * nv[:, 1]).sum(1)
            c12 = (nv[:, 1] * nv[:, 2]).sum(1)
            c20 = (nv[:, 2] * nv[:, 0]).sum(1)
            span_ok = np.minimum(np.minimum(c01, c12), c20) >= self.cos_extent
            cs = np.where(span_ok, cs, -1.0)
        return d, cs

    def midpoint_dev(self, prim, faces):
        """Area weighted mean distance of triangle edge midpoints and
        centroids: how well the surface matches the mesh *between* its
        vertices (a torus can hit both ends of a long straight strip)."""
        T = self.V[self.F[faces]]
        M = np.concatenate([(T[:, 0] + T[:, 1]) / 2, (T[:, 1] + T[:, 2]) / 2, (T[:, 2] + T[:, 0]) / 2, T.mean(1)])
        d = prim.distance(M).reshape(4, -1).max(0)
        w = self.fa[faces]
        return float((d * w).sum() / max(w.sum(), 1e-30))

    def accept(self, prim, sense, faces, tol=None):
        tol = self.tol if tol is None else tol
        d, cs = self.face_error(prim, sense, faces)
        return (d <= tol) & (cs >= self.cos_ok[faces])

    def region_points(self, faces, with_weights=False):
        """Vertices of faces with normals averaged over these faces only.

        With ``with_weights`` also returns per-vertex weights: the mesh area
        each vertex represents (a third of its incident face areas)."""
        Fs = self.F[faces]
        vid, inv = np.unique(Fs.reshape(-1), return_inverse=True)
        inv = inv.reshape(-1, 3)
        N = np.zeros((len(vid), 3))
        w = (self.fn[faces] * self.fa[faces, None])
        for k in range(3):
            np.add.at(N, inv[:, k], w)
        ln = np.linalg.norm(N, axis=1)
        ok = ln > 0
        N[ok] /= ln[ok, None]
        if not with_weights:
            return self.V[vid], N, ok
        W = np.zeros(len(vid))
        a3 = self.fa[faces] / 3.0
        for k in range(3):
            np.add.at(W, inv[:, k], a3)
        W = np.maximum(W, 1e-12 * max(W.max(), 1e-30))
        return self.V[vid], N, ok, W

    def sense_of(self, prim, faces):
        P = self.V[self.F[faces]].mean(1)
        s = (prim.normal(P) * self.fn[faces]).sum(1) @ self.fa[faces]
        return 1.0 if s >= 0 else -1.0

    def grow(self, prim, sense, seed, refit=True):
        stamp = self._new_stamp()
        st = self._stamp
        st[seed] = stamp
        region = [seed]
        frontier = seed
        total = len(seed)
        refit_at = max(2 * total, 24)
        label = self.label
        while frontier.size:
            cand = self.nb[frontier].reshape(-1)
            cand = cand[cand >= 0]
            if not cand.size:
                break
            cand = np.unique(cand)
            cand = cand[(label[cand] < 0) & (st[cand] != stamp)]
            if not cand.size:
                break
            ok = self.accept(prim, sense, cand)
            new = cand[ok]
            st[new] = stamp
            if new.size:
                region.append(new)
            frontier = new
            total += new.size
            if refit and (prim.kind != "plane" or self.refit_planes) and total >= refit_at:
                allf = np.concatenate(region)
                P, N, _, W = self.region_points(allf, True)
                p2 = prim.refit(P, N, W)
                if p2 is not None and not getattr(p2, "degenerate", lambda e: False)(self.ext):
                    okall = self.accept(p2, sense, allf)
                    if okall.mean() > 0.995:
                        prim = p2
                refit_at = 2 * total
        return np.concatenate(region), prim

    def neighbourhood(self, seed, min_verts=30, max_faces=1200, max_layers=8, min_layers=2, min_spread_deg=12.0):
        """Smoothly connected unassigned faces around a seed.

        Grows ring by ring until the patch is big enough to pin down a curved
        surface: at least ``min_layers`` rings, ``min_verts`` vertices and a
        normal spread of ``min_spread_deg`` (two rings of a torus or three
        strips of a cylinder are not enough to tell surface types apart).
        """
        stamp = self._new_stamp()
        st = self._stamp
        st[seed] = stamp
        out = [seed]
        frontier = seed
        total = len(seed)
        layers = 0
        nref = self.fn[seed[np.argmax(self.fa[seed])]]
        min_cos = 1.0
        cos_spread = np.cos(np.radians(min_spread_deg))
        while frontier.size and layers < max_layers and total < max_faces:
            layers += 1
            src = np.repeat(frontier, 3)
            cand = self.nb[frontier].reshape(-1)
            m = cand >= 0
            src, cand = src[m], cand[m]
            m = (self.label[cand] < 0) & (st[cand] != stamp)
            src, cand = src[m], cand[m]
            if not cand.size:
                break
            cs = (self.fn[src] * self.fn[cand]).sum(1)
            m = (cs >= self.cos_smooth) | self.unsure[cand] | self.unsure[src]
            cand = np.unique(cand[m])
            if not cand.size:
                break
            st[cand] = stamp
            out.append(cand)
            frontier = cand
            total += cand.size
            ok = ~self.unsure[cand]
            if ok.any():
                min_cos = min(min_cos, float((self.fn[cand[ok]] @ nref).min()))
            if layers >= min_layers and min_cos <= cos_spread:
                if len(np.unique(self.F[np.concatenate(out)])) >= min_verts:
                    break
        return np.concatenate(out)

    # --- main -----------------------------------------------------------
    def run(self):
        V, F, fn, fa = self.V, self.F, self.fn, self.fa
        tol = self.tol
        fine = min(tol * 0.05, 2e-6 * self.ext)
        sf_lab, sf_planes = planar_regions(V, F, self.topo, fn, fa, max(fine, 1e-9))
        self.sf_lab = sf_lab
        order_sf = np.argsort(sf_lab, kind="stable")
        counts = np.bincount(sf_lab)
        starts = np.r_[0, np.cumsum(counts)]
        sf_area = np.bincount(sf_lab, weights=fa)
        seeds = np.argsort(-sf_area, kind="stable")
        n_sf = len(seeds)
        rep = self.report
        deferred = []
        for k, s in enumerate(seeds):
            if k % 200 == 0:
                rep.progress(0.5 * k / max(1, n_sf))
            faces = order_sf[starts[s]:starts[s + 1]]
            faces = faces[self.label[faces] < 0]
            if not faces.size:
                continue
            if not self._process_seed(faces, allow_defer=True):
                deferred.append(s)
        # Seeds whose neighbourhood was ambiguous get a second chance once the
        # confident surfaces exist; whatever is left becomes small planes.
        for k, s in enumerate(deferred):
            if k % 200 == 0:
                rep.progress(0.5 + 0.1 * k / max(1, len(deferred)))
            faces = order_sf[starts[s]:starts[s + 1]]
            faces = faces[self.label[faces] < 0]
            if faces.size:
                self._process_seed(faces, allow_defer=False)
        self._refine()
        self._absorb_slivers()
        self._smooth_borders()
        if FIND_SWEEPS and self._find_sweeps() + self._find_fillets():
            # the pieces they leave between them and their neighbours (one
            # or two triangles each) join a surface they fit, and their
            # borders are cleaned like the others' (a tooth left in one made
            # its edge fold back, and the face was invalid)
            self._absorb_slivers()
            self._smooth_borders()
        if STRAIGHTEN and self._straighten_borders():
            # (a border moved past a small region squeezes it to a sliver
            # of a few triangles; they join a surface they fit, like the
            # pieces sweeps and fillets leave)
            self._absorb_slivers()
            self._smooth_borders()
        self._make_freeform()
        self._fix_apex_cones()
        if self.scan:
            # on a scan the later steps leave pieces that fit a neighbour
            # (a piece next to another piece had no big neighbour yet)
            for _ in range(3):
                if not self._absorb_slivers():
                    break
        return self.regions

    def _fix_apex_cones(self):
        """A cone whose region runs up to its apex without surrounding it
        (typically a long thin strip fitted by a nearly flat cone) is a
        misfit, and its boundary crosses the apex in parameter space, which
        breaks the face. Such regions become freeform cells instead."""
        regs = self.regions
        bad = []
        for ri, r in enumerate(regs):
            if r.kind != "cone":
                continue
            prim = r.prim
            P = self.V[np.unique(self.F[r.faces])]
            _, h, rad, rho = prim._parts(P)
            along = rho * np.sin(prim.alpha) + h * np.cos(prim.alpha)
            if along.min() > max(3.0 * self.tol, 0.05 * along.max()):
                continue
            # a real cone tip surrounds the apex at every distance from it
            # (near the apex any few points look like they do)
            outer = along > 0.5 * along.max()
            u, v = perp_basis(prim.d)
            ang = np.sort(np.arctan2(rad[outer] @ v, rad[outer] @ u))
            gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
            if len(ang) >= 3 and gaps.max() < np.radians(45.0):
                continue
            bad.append(ri)
        if not bad:
            return
        bad = set(bad)
        keep = [i for i in range(len(regs)) if i not in bad]
        new_regs = [regs[i] for i in keep]
        new_lab = self.label.copy()
        for k, i in enumerate(keep):
            new_lab[regs[i].faces] = k
        for i in sorted(bad):
            for cell in fit_freeform_cells(self, regs[i].faces):
                new_lab[cell.faces] = len(new_regs)
                new_regs.append(cell)
        self.regions = new_regs
        self.label = new_lab

    def _process_seed(self, faces, allow_defer=False):
        fa = self.fa
        seed_area = fa[faces].sum()
        P, N, _, W = self.region_points(faces, True)
        pl = Plane.fit(P, (self.fn[faces] * fa[faces, None]), W=W)
        best = None
        reg_p, pl = self.grow(pl, 1.0, faces, refit=self.refit_planes)
        area_p = fa[reg_p].sum()
        best = (area_p, 0, pl, 1.0, reg_p)

        nbh = self.neighbourhood(faces)
        nbh_area = fa[nbh].sum()
        hyps = []
        if nbh.size > faces.size and seed_area < 0.5 * nbh_area:
            Pn, Nn, okn, Wn = self.region_points(nbh, True)
            Pn_ok, Nn_ok, Wn_ok = Pn[okn], Nn[okn], Wn[okn]
            for cls in CURVED:
                prim = self._robust_fit(cls, Pn_ok, Nn_ok, Wn_ok)
                if prim is None:
                    continue
                sense = self.sense_of(prim, nbh)
                ok = self.accept(prim, sense, nbh)
                if ok.mean() < 0.3 or not ok[: faces.size].all():
                    continue
                explains_all = False
                if ok.all():
                    dn, _ = self.face_error(prim, sense, nbh)
                    explains_all = bool(dn.max() <= 0.1 * self.tol)
                reg, prim = self.grow(prim, sense, faces)
                # must have grown beyond the seed to be credible
                if reg.size <= faces.size:
                    continue
                p2, s2 = self.simplify(prim, sense, reg)
                if p2 is not prim:
                    if p2.kind == "plane":
                        continue  # the plane hypothesis already covers this
                    prim, sense = p2, s2
                    reg, prim = self.grow(prim, sense, faces)
                    if reg.size <= faces.size:
                        continue
                hyps.append((fa[reg].sum(), prim.rank, prim, sense, reg))
                if explains_all and nbh.size >= 3 * faces.size:
                    break  # a simpler surface already explains everything
        ext_h = self._extrusion_hypothesis(faces)
        if ext_h is not None:
            hyps.append(ext_h)
        hyps += self._axis_hypotheses(faces)
        if hyps:
            # Hypotheses that match the mesh much worse between vertices than
            # a competitor of similar size are dropped; then the largest area
            # wins, with simpler surface types preferred within 10%.
            amax = max(h[0] for h in hyps)
            q = [self.midpoint_dev(h[2], h[4]) for h in hyps]
            big = [i for i, h in enumerate(hyps) if h[0] >= 0.6 * amax]
            qmin = min(q[i] for i in big)
            hyps = [h for i, h in enumerate(hyps) if not (i in big and q[i] > max(2.5 * qmin, 0.5 * self.tol))]
            hyps.sort(key=lambda h: -h[0])
            cur = hyps[0]
            for h in hyps[1:]:
                if h[1] < cur[1] and h[0] >= 0.9 * cur[0]:
                    cur = h
            # a surface on a known axis wins a near tie: its axis is another
            # surface's, found on more of the mesh than a seed's neighbourhood
            known = [h for h in hyps if isinstance(h[2], _OnAxis)]
            if known and not isinstance(cur[2], _OnAxis):
                h = max(known, key=lambda h: h[0])
                if h[0] >= AXIS_TIE * cur[0]:
                    cur = h
            if cur[0] > 1.10 * best[0]:
                best = cur
        _, _, prim, sense, reg = best
        weak = prim.kind == "plane" and reg.size == faces.size
        if weak and allow_defer and nbh.size > faces.size:
            return False
        rid = len(self.regions)
        self.regions.append(Region(prim, reg, sense, weak))
        self.label[reg] = rid
        return True

    def _known_axes(self):
        """[(point, direction, radius)]: the axes surfaces on them have shown so
        far: stl2curves' locked cylinders, cones, tori and threads, and the
        segmenter's own once they reach AXIS_AROUND_DEG round (over a narrow
        arc a fitted axis can be far off). Gathered again only when the
        regions change."""
        if self._axes_n == len(self.regions):
            return self._axes
        out = []
        for r in self.regions:
            p = r.prim
            if p.kind == "cylinder":
                a, d, size = p.a, p.d, p.r
            elif p.kind == "cone":
                a, d, size = p.apex, p.d, None
            elif p.kind == "torus":
                a, d, size = p.c, p.d, p.R + p.r
            elif _locked(p) and getattr(getattr(p, "model", None), "kind", None) == "thread":
                a, d, size = p.model.a, p.model.d, None
            else:
                continue
            a, d = np.asarray(a, float), unit(np.asarray(d, float))
            rel = self.V[np.unique(self.F[r.faces])] - a
            rad = rel - np.outer(rel @ d, d)
            if size is None:
                size = float(np.linalg.norm(rad, axis=1).max())
            if not _locked(p):
                u, v = perp_basis(d)
                ang = np.sort(np.arctan2(rad @ v, rad @ u))
                gap = np.diff(np.r_[ang, ang[0] + 2 * np.pi]).max() if len(ang) > 1 else 2 * np.pi
                if 360.0 - np.degrees(gap) < AXIS_AROUND_DEG:
                    continue
            a = a - (a @ d) * d
            for k, (b, e, s) in enumerate(out):
                if abs(d @ e) > np.cos(np.radians(AXIS_SAME_DEG)) and np.linalg.norm(np.cross(a - b, e)) < 4 * self.tol:
                    out[k] = (b, e, max(s, size))
                    break
            else:
                out.append((a, d, size))
        self._axes, self._axes_n = out, len(self.regions)
        return out

    def _axis_hypotheses(self, faces):
        """Hypotheses like _process_seed's for surfaces about the known axes
        near the seed: a cylinder, cone, torus or sphere with only its profile
        fitted, to the seed's own corners (a chamfer's band: the free fit to
        its neighbourhood took in the faces beside it, and a plane won; and a
        band fitted with its neighbours on the axis took in a flat ring round
        it), accepted on the whole seed, and grown by the usual test with
        refits that keep the axis."""
        if not getattr(self, "known_axes", True):
            return []
        axes = self._known_axes()
        if not axes:
            return []
        c = self.V[self.F[faces]].mean((0, 1))
        near = []
        for a, d, size in axes:
            dist = float(np.linalg.norm(np.cross(c - a, d)))
            if dist <= 2 * size + AXIS_REACH * self.ext:
                near.append((dist, a, d))
        near.sort(key=lambda x: x[0])
        P, _, _, W = self.region_points(faces, True)
        out = []
        for _, a, d in near[:AXIS_NEAREST]:
            for kind in ("cylinder", "cone", "torus", "sphere"):
                p = on_axis(kind, a, d, P, W)
                if p is None or p.degenerate(self.ext):
                    continue
                s = self.sense_of(p, faces)
                if not self.accept(p, s, faces).all():
                    continue
                reg, p = self.grow(p, s, faces)
                if reg.size <= faces.size:
                    continue
                out.append((self.fa[reg].sum(), p.rank, p, s, reg))
        return out

    def _extrusion_hypothesis(self, faces, min_aspect=3.0, min_facets=3):
        """A cylinder for coarse prism-like facets (OpenSCAD cylinders and
        grooves): long, thin flat strips side by side along one direction.
        Their few vertices sit only at the strips' ends, so the usual
        neighbourhood has to grow far (into a groove's sloped ends, say)
        before it holds enough points, and the fit gets spoiled. Here the
        strips are gathered across the edges that run along their length,
        and a cylinder along that direction is fitted to them alone.
        Returns a hypothesis tuple like _process_seed's, or None."""
        V, F, fa, fn = self.V, self.F, self.fa, self.fn
        topo = self.topo
        # the direction of the seed's longest outline edge (the edges inside
        # it, like a quad's diagonal, don't count)
        he = (3 * faces[:, None] + np.arange(3)[None]).reshape(-1)
        tw = topo.twin[he]
        inside = np.zeros(len(F), dtype=bool)
        inside[faces] = True
        outline = he[(tw < 0) | ~inside[np.maximum(tw, 0) // 3]]
        if not outline.size:
            return None
        E = V[topo.he_to[outline]] - V[topo.he_from[outline]]
        L = np.linalg.norm(E, axis=1)
        k = int(np.argmax(L))
        lmax = float(L[k])
        if lmax <= 0:
            return None
        d = E[k] / lmax
        width = float(fa[faces].sum()) / lmax
        if lmax < min_aspect * max(width, 1e-12):
            return None
        sin_par = np.sin(np.radians(1.0))
        sin_perp = np.sin(np.radians(2.0))
        if np.abs(fn[faces] @ d).max() > sin_perp:
            return None
        stamp = self._new_stamp()
        st = self._stamp
        st[faces] = stamp
        strip = [faces]
        frontier = faces
        total = len(faces)
        while frontier.size and total < 4000:
            he = (3 * frontier[:, None] + np.arange(3)[None]).reshape(-1)
            tw = topo.twin[he]
            ok = tw >= 0
            he, tw = he[ok], tw[ok]
            g = tw // 3
            a, b = V[topo.he_from[he]], V[topo.he_to[he]]
            e = b - a
            el = np.maximum(np.linalg.norm(e, axis=1), 1e-30)
            parallel = np.linalg.norm(np.cross(e / el[:, None], d), axis=1) <= sin_par
            coplanar = (fn[he // 3] * fn[g]).sum(1) >= np.cos(np.radians(0.5))
            m = (parallel | coplanar) & (self.label[g] < 0) & (st[g] != stamp) & (np.abs(fn[g] @ d) <= sin_perp)
            new = np.unique(g[m])
            if not new.size:
                break
            st[new] = stamp
            strip.append(new)
            frontier = new
            total += len(new)
        strip = np.concatenate(strip)
        # at least a few distinct facets around the axis
        ang = np.round(np.degrees(np.arctan2(*(fn[strip] @ np.stack(perp_basis(d), 1)).T[::-1])), 0)
        if len(np.unique(ang)) < min_facets:
            return None
        P, N, okn, W = self.region_points(strip, True)
        try:
            prim = Cylinder.fit(P[okn], N[okn], ext=self.ext, W=W[okn], d=d)
        except Exception:
            return None
        if prim is None or prim.degenerate(self.ext):
            return None
        sense = self.sense_of(prim, strip)
        if not self.accept(prim, sense, strip).all():
            return None
        reg, prim = self.grow(prim, sense, faces)
        if reg.size <= faces.size:
            return None
        return (fa[reg].sum(), prim.rank, prim, sense, reg)

    SIMPLER = {
        "torus": ("plane", "cylinder", "sphere", "cone"),
        "cone": ("plane", "cylinder"),
        "sphere": ("plane",),
        "cylinder": ("plane",),
    }

    def simplify(self, prim, sense, faces):
        """Replace a surface by a simpler kind when that fits the same faces
        (a torus with a huge ring radius is really a cylinder, etc.)."""
        kinds = self.SIMPLER.get(prim.kind, ())
        if not kinds or _locked(prim):
            return prim, sense
        P, N, ok, W = self.region_points(faces, True)
        if isinstance(prim, _OnAxis):
            # (simpler on the same axis, or flat: a free fit would let the axis go)
            for k in kinds:
                if k == "plane":
                    p2, s2 = Plane.fit(P, (self.fn[faces] * self.fa[faces, None]), W=W), 1.0
                else:
                    p2 = on_axis(k, prim.ka, prim.kd, P, W)
                    if p2 is None or p2.degenerate(self.ext):
                        continue
                    s2 = self.sense_of(p2, faces)
                if self.accept(p2, s2, faces).all():
                    return p2, s2
            return prim, sense
        classes = {"plane": Plane, "cylinder": Cylinder, "sphere": Sphere, "cone": Cone}
        for k in kinds:
            cls = classes[k]
            if cls is Plane:
                p2 = Plane.fit(P, (self.fn[faces] * self.fa[faces, None]), W=W)
            else:
                p2 = self._robust_fit(cls, P[ok], N[ok], W[ok])
            if p2 is None:
                continue
            s2 = 1.0 if cls is Plane else self.sense_of(p2, faces)
            if self.accept(p2, s2, faces).all():
                return p2, s2
        return prim, sense

    def _robust_fit(self, cls, P, N, W=None, iters=2):
        """Weighted fit, then refit on inliers (trimmed least squares) so
        faces of a tangent neighbour (e.g. a fillet next to a hole) don't
        skew the result."""
        try:
            prim = cls.fit(P, N, ext=self.ext, W=W)
        except Exception:
            return None
        if prim is None or prim.degenerate(self.ext):
            return None
        for _ in range(iters):
            d = prim.distance(P)
            inl = d <= 2.0 * self.tol
            wsum = W.sum() if W is not None else len(P)
            winl = W[inl].sum() if W is not None else inl.sum()
            if inl.all() or inl.sum() < 8 or winl < 0.3 * wsum:
                break
            try:
                p2 = cls.fit(P[inl], N[inl], ext=self.ext, W=None if W is None else W[inl])
            except Exception:
                break
            if p2 is None or p2.degenerate(self.ext):
                break
            prim = p2
        return prim

    # --- refinement -----------------------------------------------------
    def _refine(self, iters=3):
        regs = self.regions
        F = self.F
        for it in range(iters):
            moved = 0
            lab = self.label
            # boundary faces: any neighbour with another label
            nbl = np.where(self.nb >= 0, lab[np.maximum(self.nb, 0)], -1)
            bmask = ((nbl != lab[:, None]) & (nbl >= 0)).any(1)
            bfaces = np.nonzero(bmask)[0]
            if not bfaces.size:
                break
            # error to own primitive
            own = np.full(len(bfaces), np.inf)
            best_lab = lab[bfaces].copy()
            for r in np.unique(lab[bfaces]):
                m = lab[bfaces] == r
                if regs[r].weak:
                    continue  # weak (single facet) regions dissolve readily
                d, cs = self.face_error(regs[r].prim, regs[r].sense, bfaces[m])
                own[m] = np.where(cs >= self.cos_ok[bfaces[m]], d, np.inf)
            best_err = own.copy()
            for j in range(3):
                cl = nbl[bfaces, j]
                for r in np.unique(cl[cl >= 0]):
                    m = (cl == r) & (cl != lab[bfaces])
                    if not m.any():
                        continue
                    reg = regs[r]
                    if reg.weak:
                        continue
                    d, cs = self.face_error(reg.prim, reg.sense, bfaces[m])
                    e = np.where(cs >= self.cos_ok[bfaces[m]], d, np.inf)
                    e = np.where(e <= self.tol, e, np.inf)
                    idx = np.nonzero(m)[0]
                    better = e < best_err[idx] - 0.05 * self.tol
                    best_err[idx[better]] = e[better]
                    best_lab[idx[better]] = r
            ch = best_lab != lab[bfaces]
            # weak regions dissolve readily into neighbours that accept them
            if ch.any():
                self.label[bfaces[ch]] = best_lab[ch]
                moved = int(ch.sum())
            if moved == 0:
                break
        self._rebuild_regions()

    def _absorb_slivers(self, max_tris=3, tol_factor=3.0):
        """Tiny regions (slivers where three surfaces meet) join the
        neighbouring surface that fits them best. Genuine tiny faces, such
        as the corner triangle of a chamfered box, fit no neighbour and stay."""
        regs = self.regions
        lab = self.label
        changed = False
        cos_relaxed = np.cos(np.radians(60.0))
        if self.scan:
            # (noise splits a scan into pieces of many small triangles)
            max_tris = max(max_tris, 50)
        areas = np.array([self.fa[r.faces].sum() for r in regs])
        big = areas[np.array([len(r.faces) > max_tris for r in regs], dtype=bool)]
        small_area = 0.02 * (np.median(big) if len(big) else areas.max())
        if self.scan:
            # on a scan most regions are small pieces, so their median says
            # nothing about what is small: hundreds of 1-3 triangle pieces
            # stayed, each punching a hole in the face around it (a scanned
            # plate's top face got 334 holes and was invalid)
            small_area = max(small_area, 1e-4 * float(self.fa.sum()))
        # regions smaller than a face can be (see _too_small_for_face) are
        # slivers too, however many triangles they have: a strip of 18
        # needle triangles (0.01 mm2) on a printed mount stuck into the
        # cylinder beside it, whose outline then ran in and straight back
        # out, crossed itself, and the face was invalid
        tiny = areas <= 25.0 * self.tol ** 2
        few = np.array([len(r.faces) <= max_tris and (r.weak or areas[ri] <= small_area)
                        for ri, r in enumerate(regs)], dtype=bool)
        sliver = [ri for ri, r in enumerate(regs) if len(r.faces) and (few[ri] or tiny[ri])]
        # (those that are only small in area join a surface they lie on
        # within the tolerance: a finely meshed worm gear's small true faces
        # sit a few tolerances off their neighbours, and taking them in cost
        # it accuracy)
        reach = np.where(few, tol_factor, 1.0) * self.tol
        # (a few passes: a triangle inside a strip joins once its neighbour has)
        for _ in range(4):
            moved = False
            for ri in sliver:
                for f in regs[ri].faces:
                    if lab[f] != ri:
                        continue
                    best, best_e = None, np.inf
                    for g in self.nb[f]:
                        if g < 0:
                            continue
                        rj = lab[g]
                        if rj == ri or regs[rj].weak or len(regs[rj].faces) <= max_tris or tiny[rj]:
                            continue
                        d, cs = self.face_error(regs[rj].prim, regs[rj].sense, np.array([f]))
                        # (on a scan a piece this small that sits on a surface
                        # is noise, whichever way its triangles point)
                        if cs[0] < cos_relaxed and not (self.unsure[f] or self.needle[f] or self.scan):
                            continue
                        if d[0] <= reach[ri] and d[0] < best_e:
                            best, best_e = rj, d[0]
                    if best is not None:
                        lab[f] = best
                        moved = changed = True
            if not moved:
                break
        if changed:
            self._rebuild_regions()
        return changed

    def _gap(self, prim, faces):
        """How far triangles stray from a surface: the largest distance of
        their corners, edge midpoints and centre."""
        if len(faces) == 0:
            return np.zeros(0)
        T = self.V[self.F[faces]]
        P = np.concatenate([T, (T + np.roll(T, -1, axis=1)) / 2, T.mean(1, keepdims=True)], axis=1)
        return prim.distance(P.reshape(-1, 3)).reshape(len(faces), -1).max(1)

    def _smooth_borders(self, passes=6):
        """Triangles that stick out of their surface into a neighbour's (a
        "tooth": two or all three of its neighbours belong to one other
        surface) join that surface when they fit it. A CAD export's long thin
        triangles otherwise leave borders that run out and straight back,
        and the edge curves drawn along them fold over themselves, which
        makes the faces invalid. Returns whether anything changed."""
        regs, lab, nb = self.regions, self.label, self.nb
        # (swept surfaces and fillets too, which have a motion attribute:
        # they are exact, like these)
        analytic = np.array([not r.weak and (r.kind in ("plane", "cylinder", "cone", "sphere", "torus")
                                             or hasattr(r.prim, "motion"))
                             for r in regs], dtype=bool)
        changed = False
        for _ in range(passes):
            nl = np.where(nb >= 0, lab[np.maximum(nb, 0)], -1)
            other = (nl >= 0) & (nl != lab[:, None])
            cand = np.nonzero(other.sum(1) >= 2)[0]
            if not len(cand):
                break
            # the label that two or three of the neighbours share
            a, b_, c = nl[cand, 0], nl[cand, 1], nl[cand, 2]
            maj = np.where((a == b_) | (a == c), a, np.where(b_ == c, b_, -1))
            ok = (maj >= 0) & (maj != lab[cand])
            cand, maj = cand[ok], maj[ok]
            ok = analytic[maj]
            cand, maj = cand[ok], maj[ok]
            # a thin triangle along a border can fit both surfaces (it lies
            # on the line where a fillet meets a wall): it goes to the one it
            # follows more closely, so moving it can't be undone next pass
            gap_new = np.full(len(cand), np.inf)
            gap_own = np.full(len(cand), np.inf)
            for rj in np.unique(maj):
                k = np.nonzero(maj == rj)[0]
                fit = self.accept(regs[rj].prim, regs[rj].sense, cand[k])
                gap_new[k[fit]] = self._gap(regs[rj].prim, cand[k[fit]])
            for ri in np.unique(lab[cand]):
                k = np.nonzero(lab[cand] == ri)[0]
                gap_own[k] = self._gap(regs[ri].prim, cand[k])
            move = gap_new < gap_own
            if not move.any():
                break
            lab[cand[move]] = maj[move]
            changed = True
        if changed:
            self._rebuild_regions()
        return changed

    def _straighten_borders(self, rings=8):
        """Where two exact surfaces meet smoothly (a fillet running into a
        wall), the triangles along their border fit both, and which one
        each joined was close to chance: the border zigzagged along the
        mesh, and the faces came out with saw-tooth edges (on a printed
        mount, 82 of 479 edges). Triangles within ``rings`` of such a
        border that fit both surfaces may change sides, and they take the
        sides that make the border shortest (a minimum cut). Triangles that
        fit only one surface stay where they are, so no face moves further
        from the mesh. Returns whether anything changed."""
        from scipy import sparse
        from scipy.sparse.csgraph import maximum_flow

        regs, lab, nb = self.regions, self.label, self.nb
        topo = self.topo
        analytic = np.array([not r.weak and (r.kind in ("plane", "cylinder", "cone", "sphere", "torus")
                                             or hasattr(r.prim, "motion"))
                             for r in regs], dtype=bool)
        L = np.linalg.norm(self.V[topo.he_to] - self.V[topo.he_from], axis=1).reshape(-1, 3)
        unit = max(float(np.median(L)), 1e-12)
        nl = np.where(nb >= 0, lab[np.maximum(nb, 0)], -1)
        cross = (nl >= 0) & (nl != lab[:, None])
        f_, k_ = np.nonzero(cross)
        a_, b_ = lab[f_], nl[f_, k_]
        keep = (a_ < b_) & analytic[a_] & analytic[b_]
        pairs = np.unique(np.stack([a_[keep], b_[keep]], 1), axis=0) if keep.any() else np.zeros((0, 2), int)
        changed = False
        self._pieces_before = {}
        for r in np.unique(pairs):
            self._pieces_before[int(r)] = self._pieces(lab == r)
        for a, b in pairs:
            # the triangles of either surface near their border
            on = (lab == a) | (lab == b)
            front = np.unique(np.r_[f_[(a_ == a) & (b_ == b)], f_[(a_ == b) & (b_ == a)]])
            front = front[on[front]]
            seen = np.zeros(len(lab), dtype=bool)
            seen[front] = True
            for _ in range(rings):
                n = nb[front].reshape(-1)
                n = np.unique(n[n >= 0])
                n = n[on[n] & ~seen[n]]
                if not len(n):
                    break
                seen[n] = True
                front = n
            near = np.nonzero(seen)[0]
            free = near[self.accept(regs[a].prim, regs[a].sense, near)
                        & self.accept(regs[b].prim, regs[b].sense, near)]
            if len(free) < 2:
                continue
            # graph: the free triangles, plus a source (side a) and a sink
            # (side b) standing for the triangles that stay
            idx = np.full(len(lab), -1)
            idx[free] = np.arange(len(free))
            src, snk = len(free), len(free) + 1
            rows, cols, caps = [], [], []
            # how well each surface fits each free triangle: moving one to the surface it
            # fits worse costs FIT_WEIGHT x (distance / tolerance)^2 (a typical edge of
            # border weighs 100), so the border keeps triangles on their better surface
            # (both P1S braces: rows went to the surface they fit worse, -1.2% and -1.8%
            # volume, while the mirror image came out right)
            da = self.face_error(regs[a].prim, regs[a].sense, free)[0] / max(self.tol, 1e-12)
            db = self.face_error(regs[b].prim, regs[b].sense, free)[0] / max(self.tol, 1e-12)
            on_a = np.round(FIT_WEIGHT * np.square(np.clip(db, 0, 1))).astype(np.int64)   # (cut if it goes to b)
            on_b = np.round(FIT_WEIGHT * np.square(np.clip(da, 0, 1))).astype(np.int64)   # (cut if it goes to a)
            ia = np.nonzero(on_a > 0)[0]
            ib = np.nonzero(on_b > 0)[0]
            rows += [np.full(len(ia), src), ib]
            cols += [ia, np.full(len(ib), snk)]
            caps += [on_a[ia], on_b[ib]]
            for k in range(3):
                o = nb[free, k]
                w = np.maximum(1, np.round(100 * L[free, k] / unit)).astype(np.int64)
                ok = o >= 0
                inner = ok & (idx[np.maximum(o, 0)] >= 0)
                rows.append(idx[free[inner]])
                cols.append(idx[o[inner]])
                caps.append(w[inner])
                for side, node in ((a, src), (b, snk)):
                    fixed = ok & ~inner & (lab[np.maximum(o, 0)] == side)
                    i = idx[free[fixed]]
                    if node == src:
                        rows.append(np.full(len(i), src))
                        cols.append(i)
                    else:
                        rows.append(i)
                        cols.append(np.full(len(i), snk))
                    caps.append(w[fixed])
            rows, cols, caps = np.concatenate(rows), np.concatenate(cols), np.concatenate(caps)
            if not ((rows == src).any() and (cols == snk).any()):
                continue  # one surface is all band: nothing holds it in place
            n = len(free) + 2
            G = sparse.coo_matrix((caps.astype(np.int32), (rows, cols)), shape=(n, n)).tocsr()
            G.sum_duplicates()
            flow = maximum_flow(G, src, snk).flow
            # what the source still reaches through unsaturated links is
            # side a; the rest is side b
            R = (G - flow).tocsr()
            R.data[R.data < 0] = 0
            R.eliminate_zeros()
            # (a reverse link carries back what flowed forward)
            back = flow.T.tocsr()
            back.data[back.data < 0] = 0
            R = (R + back).tocsr()
            R.eliminate_zeros()
            reach = np.zeros(n, dtype=bool)
            reach[src] = True
            front_ = np.array([src])
            while len(front_):
                nxt = np.unique(R[front_].indices)
                nxt = nxt[~reach[nxt]]
                reach[nxt] = True
                front_ = nxt
            # Several borders can be equally short (two parallel rows along a
            # fillet): what the source reaches is the shortest border nearest
            # side a, which side is a only by region number, so the same part
            # mirrored got another border. Every node that can't reach the sink
            # is the one nearest side b; the triangles between the two (tied)
            # go together to the side _tied_side picks from their shape.
            RT = R.T.tocsr()
            to_snk = np.zeros(n, dtype=bool)
            to_snk[snk] = True
            front_ = np.array([snk])
            while len(front_):
                nxt = np.unique(RT[front_].indices)
                nxt = nxt[~to_snk[nxt]]
                to_snk[nxt] = True
                front_ = nxt
            tied = np.nonzero(~reach[:len(free)] & ~to_snk[:len(free)])[0]
            if len(tied) and self._tied_side(free[tied], a, b) == a:
                reach[tied] = True
            new = np.where(reach[:len(free)], a, b)
            if not (new != lab[free]).any():
                continue
            old = lab[free].copy()
            lab[free] = new
            # a cut that leaves either surface in pieces (a thin one cut
            # through) is undone: the pieces made faces that failed
            if any(self._pieces(lab == r) > self._pieces_before.get(r, 1) for r in (a, b)):
                lab[free] = old
                continue
            # a cut that squeezes either surface into a strip is undone too:
            # on a wheel rim one 3.7 tolerances wide became 1.3, and the
            # strip's face got triangles at the origin in a slicer's mesher
            after = {r: self._width(lab == r) for r in (a, b)}
            lab[free] = old
            before = {r: self._width(lab == r) for r in (a, b)}
            if any(after[r] < 2 * self.tol and after[r] < before[r] for r in (a, b)):
                continue
            lab[free] = new
            changed = True
        if changed:
            # (no refits or merges: the surfaces haven't changed)
            self._rebuild_regions(refit=False, merge=False)
        return changed

    # cosine: tied triangles' mean agreement with two surfaces' normals closer than this
    # is a tie (a fillet row 11 degrees off a wall differs by 0.02)
    TIE_COS = 1e-4
    # share of the larger region's area: regions closer in area than this are a tie
    TIE_AREA = 1e-6

    def _tied_side(self, faces, a, b):
        """Which of regions a and b takes triangles that fit both, when giving
        them to either leaves an equally short border; from the shape alone, so
        the answer doesn't depend on the order of triangles or regions: the
        surface their directions agree with better (area weighted); then a curved
        surface over a flat one (the border then runs along the tangent line,
        which lies on both); then the larger region; then the region whose
        area-weighted centre comes first by x, then y, then z."""
        regs = self.regions
        w = self.fa[faces]
        cen = self.V[self.F[faces]].mean(1)
        agree = []
        for r in (a, b):
            nrm = regs[r].prim.normal(cen) * regs[r].sense
            agree.append(float(((self.fn[faces] * nrm).sum(1) * w).sum() / max(w.sum(), 1e-300)))
        if abs(agree[0] - agree[1]) > self.TIE_COS:
            return a if agree[0] > agree[1] else b
        flat = [regs[r].kind == "plane" for r in (a, b)]
        if flat[0] != flat[1]:
            return b if flat[0] else a
        area = [float(self.fa[regs[r].faces].sum()) for r in (a, b)]
        if abs(area[0] - area[1]) > self.TIE_AREA * max(area):
            return a if area[0] > area[1] else b
        c = []
        for r in (a, b):
            fr = regs[r].faces
            wr = self.fa[fr]
            c.append(tuple(np.round((self.V[self.F[fr]].mean(1) * wr[:, None]).sum(0) / max(wr.sum(), 1e-300), 9)))
        return a if c[0] <= c[1] else b

    def _width(self, mask):
        """Mean width of a set of triangles: twice their area over the
        length of their outline."""
        faces = np.nonzero(mask)[0]
        if not len(faces):
            return np.inf
        he = (3 * faces[:, None] + np.arange(3)[None]).reshape(-1)
        twin = self.topo.twin[he]
        outer = (twin < 0) | ~mask[np.maximum(twin, 0) // 3]
        a = self.V[self.topo.he_from[he[outer]]]
        b = self.V[self.topo.he_to[he[outer]]]
        perim = float(np.linalg.norm(b - a, axis=1).sum())
        return 2.0 * float(self.fa[faces].sum()) / perim if perim > 0 else np.inf

    def _pieces(self, mask):
        """How many separate pieces the triangles in ``mask`` make."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        f = np.nonzero(mask)[0]
        if not len(f):
            return 0
        idx = np.full(len(mask), -1)
        idx[f] = np.arange(len(f))
        n = self.nb[f]
        ok = (n >= 0) & mask[np.maximum(n, 0)]
        r = np.repeat(np.arange(len(f)), 3)[ok.reshape(-1)]
        c = idx[n[ok]]
        G = sparse.coo_matrix((np.ones(len(r)), (r, c)), shape=(len(f), len(f)))
        return connected_components(G, directed=False)[0]

    def _flat_slivers_to_long_edge(self):
        """Give each triangle with no area to speak of (its corners in a line) to the
        region across its longest edge: it has no surface of its own, and held by a
        short edge it makes a spike of zero width in its face's outline, whose two
        sides OpenCASCADE finds crossing (Solar Panel Bracket Large: two such spikes
        broke a plane and a freeform face, and the whole body was saved faceted).
        Returns True if anything changed."""
        lab, nb, topo = self.label, self.nb, self.topo
        L = np.linalg.norm(self.V[topo.he_to] - self.V[topo.he_from], axis=1).reshape(-1, 3)
        k = L.argmax(1)
        rows = np.arange(len(lab))
        h = 2.0 * self.fa / np.maximum(L[rows, k], 1e-300)
        o = nb[rows, k]
        changed = False
        # (a fan of them along one long edge passes the region on one by one)
        for _ in range(3):
            move = np.nonzero((h < SLIVER_HEIGHT * self.tol) & (o >= 0) & (lab[np.maximum(o, 0)] != lab))[0]
            if not len(move):
                break
            lab[move] = lab[o[move]]
            changed = True
        if changed:
            # (the surfaces stay as they are and no regions merge)
            self._rebuild_regions(refit=False, merge=False)
        return changed

    def _absorb_isolated_fragments(self):
        """Absorb triangles that are completely surrounded by other regions.

        A triangle with no same-region neighbour creates a disconnected face
        component that OpenCASCADE cannot represent as a single valid face.
        This arises when split_pinched_regions re-labels some triangles and
        leaves tiny fragments behind.  Each such triangle is merged into
        whichever neighbour has the lowest gap to its surface; ties are broken
        by region size (larger region = more stable boundary).

        Does not run the majority pass or touch any other region boundaries.
        Returns True if anything changed.
        """
        regs, lab, nb = self.regions, self.label, self.nb
        nl = np.where(nb >= 0, lab[np.maximum(nb, 0)], -1)
        other = (nl >= 0) & (nl != lab[:, None])
        isolated = np.nonzero(other.all(1) & (nl >= 0).any(1))[0]
        if not len(isolated):
            return False
        best_dst = np.full(len(isolated), -1, dtype=np.int64)
        gap_best = np.full(len(isolated), np.inf)
        sizes = np.array([len(r.faces) for r in regs])
        for edge in range(3):
            dst = nl[isolated, edge]
            valid = dst >= 0
            for rj in np.unique(dst[valid]):
                k = np.nonzero(valid & (dst == rj))[0]
                g = self._gap(regs[rj].prim, isolated[k])
                for ki, idx in enumerate(k):
                    better = g[ki] < gap_best[idx] - 1e-10
                    same_gap = abs(g[ki] - gap_best[idx]) < 1e-10
                    bigger = best_dst[idx] < 0 or sizes[rj] > sizes[best_dst[idx]]
                    if better or (same_gap and bigger):
                        gap_best[idx] = g[ki]
                        best_dst[idx] = rj
        # Only absorb if the destination surface actually fits the fragment
        # within tolerance.  A fragment with gap > tol to all neighbours is
        # genuinely ambiguous — leave it for the invalid-face demotion path
        # rather than forcing it into a region whose surface it doesn't match.
        move = (best_dst >= 0) & (gap_best <= self.tol)
        if not move.any():
            return False
        lab[isolated[move]] = best_dst[move]
        # (the surfaces stay as they are and no regions merge: two
        # neighbouring cylinders of a worm gear merged into one that
        # followed neither closely, between the build's rounds)
        self._rebuild_regions(refit=False, merge=False)
        return True

    def _absorb_self_crossing_strips(self, sources, passes=4):
        """Try to remove border triangles from faces whose *built* wire crosses.

        This is deliberately not part of normal segmentation.  It is a
        second, topology-checked attempt after OpenCASCADE has proved that a
        face's wire crosses itself.  A candidate triangle must share one edge
        with a neighbouring analytic surface (so at least one of that edge's
        two vertices already participates in a face of the neighbour), and the
        whole triangle must fit the neighbour within tolerance and better than
        its current surface.  The caller builds the result again and keeps it
        only if the reported self-crossing actually decreases without adding
        invalid faces.  The purely geometric criterion (shared-edge anchor +
        gap improvement) is model-agnostic: it fires wherever a region boundary
        produces a crossing wire, regardless of the model.
        """
        regs, lab, nb, F = self.regions, self.label, self.nb, self.F
        sources = set(sources)
        if not sources:
            return False
        nr = len(regs)
        target = np.array([not r.weak and (r.kind in ("plane", "cylinder", "cone", "sphere", "torus")
                                           or hasattr(r.prim, "motion")) for r in regs], dtype=bool)
        changed = False
        for _ in range(passes):
            nl = np.where(nb >= 0, lab[np.maximum(nb, 0)], -1)
            touch = np.unique(F.reshape(-1).astype(np.int64) * nr + np.repeat(lab, 3))
            cand = np.nonzero(np.isin(lab, tuple(sources)) & (nl != lab[:, None]).any(1))[0]
            best = np.full(len(cand), -1)
            for edge in range(3):
                dst = nl[cand, edge]
                ok = (dst >= 0) & (dst != lab[cand]) & target[np.maximum(dst, 0)] & (best < 0)
                # A candidate triangle is anchored at the boundary when at
                # least one vertex of the shared edge (the edge that touches
                # the neighbour) already participates in a face of that
                # neighbour.  Half-edge convention: nb[fi, e] is the
                # neighbour across the edge from F[fi, e] to F[fi, (e+1)%3].
                dmax = np.maximum(dst, 0)
                v0 = F[cand, edge].astype(np.int64) * nr + dmax
                v1 = F[cand, (edge + 1) % 3].astype(np.int64) * nr + dmax
                ok &= np.isin(v0, touch) | np.isin(v1, touch)
                best[ok] = dst[ok]
            keep = best >= 0
            cand, best = cand[keep], best[keep]
            if not len(cand):
                break
            gap_new = np.full(len(cand), np.inf)
            gap_own = np.full(len(cand), np.inf)
            for rj in np.unique(best):
                k = np.nonzero(best == rj)[0]
                fit = self.accept(regs[rj].prim, regs[rj].sense, cand[k])
                gap_new[k[fit]] = self._gap(regs[rj].prim, cand[k[fit]])
            for ri in np.unique(lab[cand]):
                k = np.nonzero(lab[cand] == ri)[0]
                gap_own[k] = self._gap(regs[ri].prim, cand[k])
            move = (gap_new <= self.tol) & (gap_new < gap_own)
            if not move.any():
                break
            lab[cand[move]] = best[move]
            changed = True
        if changed:
            self._rebuild_regions()
        return changed

    def _find_sweeps(self, sharp_deg=30.0, min_regions=3):
        """Smooth patches (bounded by sharp edges) that one swept surface
        explains (see sweep.py: a helical gear's tooth flank, a thread, a
        straight gear's tooth side) become one face, instead of the many
        small planes, cylinders and freeform pieces that each follow a bit
        of it. Patches already one good analytic surface are left alone.
        Returns the number of patches replaced."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        from .sweep import MAX_RMS_DEG, fit_motion, fit_swept, motion_rms

        F, fn, fa, nb, lab = self.F, self.fn, self.fa, self.nb, self.label
        nf = len(F)
        rows = np.repeat(np.arange(nf), 3)
        cols = nb.reshape(-1)
        m = cols >= 0
        smooth = m.copy()
        smooth[m] = (fn[rows[m]] * fn[cols[m]]).sum(1) >= np.cos(np.radians(sharp_deg))
        G = sparse.coo_matrix((np.ones(smooth.sum(), np.int8), (rows[smooth], cols[smooth])), shape=(nf, nf))
        ncomp, comp = connected_components(G, directed=False)
        total = float(fa.sum())
        order = np.argsort(comp, kind="stable")
        starts = np.r_[0, np.nonzero(np.diff(comp[order]))[0] + 1, nf]
        cen = self.V[F].mean(1)
        found = 0
        kinds, sure = {}, 1.0
        cands = []
        for k in range(len(starts) - 1):
            faces = order[starts[k]:starts[k + 1]]
            # (no limit on area: a ring gear's flanks are each a tiny share
            # of the part)
            if len(faces) < 12 or fa[faces].sum() < 1e-5 * total:
                continue
            regs = np.unique(lab[faces])
            if len(regs) < min_regions:
                continue  # already one or two faces: nothing to gain
            if any(_locked(self.regions[r].prim) for r in regs):
                continue  # (stl2curves' surfaces stay: see _locked)
            # one good analytic surface covering nearly all of it stays
            areas = np.bincount(np.searchsorted(regs, lab[faces]), weights=fa[faces])
            top = self.regions[regs[int(np.argmax(areas))]]
            if areas.max() > 0.95 * areas.sum() and not top.weak and top.kind != "freeform":
                continue
            cands.append(faces)
        # The patches of one part often share one motion (all flanks of a
        # gear screw about its axis): fitted on all of them at once it is
        # far better determined than on one thin flank (whose normals barely
        # vary), and the flanks come out exactly alike. Groups of patches
        # that agree on a motion get it; the rest use their own.
        own = [fit_motion(cen[f], fn[f], fa[f]) for f in cands]
        # a patch no one motion explains may hold parts that one does (a
        # filleted extrusion: side, fillet and top are one smooth patch)
        split = []
        for f, m in zip(cands, own):
            subs = self._sub_patches(f, min_regions) if m is None else []
            split += subs if subs else [(f, m)]
        cands = [f for f, _ in split]
        own = [m for _, m in split]
        motion = list(own)
        left = set(range(len(cands)))
        area = [float(fa[f].sum()) for f in cands]

        def agree(mot):
            return [i for i in left if motion_rms(mot, cen[cands[i]], fn[cands[i]], fa[cands[i]]) <= MAX_RMS_DEG]

        def fit_on(idx):
            u = np.concatenate([cands[i] for i in idx])
            return fit_motion(cen[u], fn[u], fa[u])

        while len(left) >= 2:
            hyps = []
            swept_like = [i for i in left if own[i] is not None and own[i].kind != "revolution"]
            if len(swept_like) >= 2:
                hyps.append(fit_on(swept_like))
            big = sorted((i for i in left if own[i] is not None), key=lambda i: -area[i])[:20]
            hyps += [own[i] for i in big]
            # a thin flank alone can't tell its motion for sure (its normals
            # barely vary), but it can propose one: kept only when other
            # patches agree, and then refitted on all of them
            unsure = sorted((i for i in left if own[i] is None), key=lambda i: -area[i])[:20]
            hyps += [fit_motion(cen[cands[i]], fn[cands[i]], fa[cands[i]], min_gap=1.2) for i in unsure]
            best, best_area = None, 0.0
            for h in hyps:
                if h is None:
                    continue
                g = agree(h)
                ga = sum(area[i] for i in g)
                if len(g) >= 2 and ga > best_area:
                    best, best_area, group = h, ga, g
            if best is None:
                break
            ref = fit_on(group)
            if ref is None:
                # the group together must be sure of it (a proposal from a
                # thin patch alone is not enough)
                for i in group:
                    left.discard(i)
                continue
            group = agree(ref) or group
            for i in group:
                motion[i] = ref
            left -= set(group)
        for i, faces in enumerate(cands):
            vid = np.unique(F[faces])
            mot = motion[i]
            if mot is None:
                continue
            if True:
                # one cone, cylinder, sphere or torus, when it fits, is
                # better than a sweep: a turned straight line or arc is one
                # (a faceted chamfer has corners only at its two ends, too
                # few for a profile curve), so are lines through one point
                # along a circle, and pieces of one cylinder that a patch
                # holds (extruded, they followed a printed liner 2.5 times
                # less closely than the cylinder)
                prim = self._revolved_analytic(faces)
                if prim is not None:
                    lab[faces] = len(self.regions)
                    self.regions.append(Region(prim, faces, self.sense_of(prim, faces)))
                    found += 1
                    continue
            surf, dev = fit_swept(cen[faces], fn[faces], fa[faces], self.tol, X=self.V[vid], motion=mot)
            if surf is None or dev > self.tol:
                continue
            mot = surf.motion
            # all the way round (a thread): not yet. A turned face short of
            # a full turn is one face (the rounded rim round a hole, broken
            # only where an arm leaves it), as long as its range, margins
            # included, doesn't come back over itself
            most = 1.95 * np.pi if mot.kind == "revolution" else 1.6 * np.pi
            if mot.kind != "extrusion" and mot.turns(surf.t_range[1] - surf.t_range[0]) > most:
                continue
            # every triangle of it on the surface, between its corners too
            T = self.V[F[faces]]
            M = np.concatenate([T, (T + np.roll(T, -1, axis=1)) / 2], axis=1).reshape(-1, 3)
            if surf.distance(M).max() > self.tol_mid:
                continue
            if not self._no_worse(surf, faces):
                continue
            sense = self.sense_of(surf, faces)
            faces = self._take_in_pieces(surf, faces)
            if surf.curve.closed:
                # all the way round: two faces of half the profile each
                half = surf.half_of(cen[faces])
                if half.all() or not half.any():
                    continue
                for hs, hf in zip(surf.halves(), (faces[half], faces[~half])):
                    lab[hf] = len(self.regions)
                    self.regions.append(Region(hs, hf, self.sense_of(hs, hf)))
            else:
                lab[faces] = len(self.regions)
                self.regions.append(Region(surf, faces, sense))
            found += 1
            kinds[mot.kind] = kinds.get(mot.kind, 0) + 1
            sure = min(sure, mot.sureness)
        if kinds:
            words = {"helix": "helical", "extrusion": "extruded", "revolution": "turned", "taper": "tapered"}
            self.report.log("Found " + ", ".join(f"{n} {words[k]}" for k, n in kinds.items())
                            + f" surface(s) swept along one motion (sureness {sure:.0%})")
        if found:
            self._rebuild_regions()
        return found

    def _find_fillets(self, min_regions=3):
        """Fillets along curved edges (see fillet.py): neighbouring tori and
        cylinders of one radius, which the segmenter cut a rounded edge
        into, become one pipe face when every triangle lies on it. Pieces
        that are all one torus or cylinder already are left alone. Returns
        the number of fillets found."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        from .fillet import fit_pipe

        regs, lab, nb, F, fa = self.regions, self.label, self.nb, self.F, self.fa
        nr, nf = len(regs), len(F)
        rad = np.full(nr, np.nan)
        for i, reg in enumerate(regs):
            if reg.weak or not len(reg.faces) or _locked(reg.prim):
                continue
            if reg.kind in ("torus", "cylinder") and not hasattr(reg.prim, "motion"):
                rad[i] = reg.prim.r
        rows = np.repeat(np.arange(nf), 3)
        cols = nb.reshape(-1)
        m = cols >= 0
        a, b = lab[rows[m]], lab[cols[m]]
        k = (a != b) & (a >= 0) & (b >= 0)
        a, b = a[k], b[k]
        k = np.isfinite(rad[a]) & np.isfinite(rad[b])
        a, b = a[k], b[k]
        same = np.abs(rad[a] - rad[b]) <= np.maximum(0.03 * np.minimum(rad[a], rad[b]), 0.5 * self.tol)
        G = sparse.coo_matrix((np.ones(same.sum(), np.int8), (a[same], b[same])), shape=(nr, nr))
        ncomp, comp = connected_components(G, directed=False)
        cen = self.V[F].mean(1)
        found = 0
        for c in range(ncomp):
            members = np.nonzero(comp == c)[0]
            if len(members) < min_regions:
                continue
            p0 = regs[members[0]].prim
            if all(regs[i].prim.kind == p0.kind and regs[i].prim.same_as(p0, self.tol, self.ext) for i in members[1:]):
                continue
            faces = np.concatenate([np.asarray(regs[i].faces) for i in members])
            # spine samples: each triangle's centre moved onto its piece's
            # spine (a torus's centre circle, a cylinder's axis)
            C = []
            for i in members:
                p, P = regs[i].prim, cen[regs[i].faces]
                C.append(p._spine(P) if p.kind == "torus" else p.a + np.outer((P - p.a) @ p.d, p.d))
            w = np.array([fa[regs[i].faces].sum() for i in members])
            r = float(rad[members][np.argsort(rad[members])][np.searchsorted(np.cumsum(w[np.argsort(rad[members])]), 0.5 * w.sum())])
            C = np.vstack(C)
            surf, dev = fit_pipe(C, self.V[np.unique(F[faces])], r, self.tol)
            # the fillet's other pieces (the segmenter fits some as cones or
            # freeform cells): small neighbours near the pipe join, and it is
            # fitted again on all of them
            for _ in range(2):
                if surf is None:
                    break
                inside = np.zeros(nf, dtype=bool)
                inside[faces] = True
                near = nb[faces]
                near = np.unique(lab[near[near >= 0]])
                more = []
                for l in near:
                    if l < 0 or hasattr(regs[l].prim, "motion"):
                        continue
                    g = np.asarray(regs[l].faces)
                    g = g[(lab[g] == l) & ~inside[g]]
                    if len(g) and len(g) <= 0.2 * len(faces) and self._gap(surf, g).max() <= 3 * self.tol:
                        more.append(g)
                if not more:
                    break
                g = np.concatenate(more)
                faces = np.concatenate([faces, g])
                C = np.vstack([C, surf.curve.at(surf.foot(cen[g])[0])])
                surf, dev = fit_pipe(C, self.V[np.unique(F[faces])], r, self.tol)
            if surf is None or dev > self.tol:
                continue
            T = self.V[F[faces]]
            M = np.concatenate([T, (T + np.roll(T, -1, axis=1)) / 2], axis=1).reshape(-1, 3)
            if surf.distance(M).max() > self.tol_mid or not self._no_worse(surf, faces):
                continue
            faces = self._take_in_pieces(surf, faces)
            if surf.curve.closed:
                halves = surf.halves()
                half = surf.half_of(cen[faces])
                if halves is None or half.all() or not half.any():
                    continue
                for hs, hf in zip(halves, (faces[half], faces[~half])):
                    lab[hf] = len(self.regions)
                    self.regions.append(Region(hs, hf, self.sense_of(hs, hf)))
            else:
                lab[faces] = len(self.regions)
                self.regions.append(Region(surf, faces, self.sense_of(surf, faces)))
            found += 1
        if found:
            self.report.log(f"Found {found} fillet(s) along curved edges (a ball rolled along the edge)")
            self._rebuild_regions()
        return found

    def _sub_patches(self, faces, min_regions=3, max_deg=3.0, min_tris=30):
        """Parts of a smooth patch that one motion explains, as [(faces,
        motion)]: the side of an extrusion whose top edge is rounded (the
        fillet joins side and top smoothly, so the three are one patch). The
        patch's own analytic regions propose the motions: a cylinder its
        axis to slide along or turn about, a cone its axis to turn about. The triangles square to a motion (within max_deg), in
        connected pieces over at least min_regions regions, are confirmed by
        fitting the motion on them alone: it must come out the motion
        proposed."""
        from .sweep import Motion, fit_motion

        F, fn, fa, nb, lab = self.F, self.fn, self.fa, self.nb, self.label
        regs = np.unique(lab[faces])
        props = []
        for r in regs:
            p = self.regions[r].prim
            w = float(fa[self.regions[r].faces].sum())
            if p.kind == "cylinder":
                props += [(w, Motion("extrusion", p.a, p.d)), (w, Motion("revolution", p.a, p.d))]
            elif p.kind == "cone":
                props.append((w, Motion("revolution", p.apex, p.d)))
            # (not a torus: inside a patch no one motion explains, those are
            # nearly always pieces of a fillet, which _find_fillets joins; a
            # motion proposed by one turned a fillet into a helix)
        # (extrusions along one direction are one motion, whatever cylinder
        # proposed them)
        seen, hyps = [], []
        for w, m in sorted(props, key=lambda x: -x[0]):
            key = (m.kind, m.d, m.a)
            if any(k == m.kind and abs(d @ m.d) > np.cos(np.radians(1.0))
                   and (k == "extrusion" or np.linalg.norm(np.cross(a - m.a, m.d)) < self.tol)
                   for k, d, a in seen):
                continue
            seen.append(key)
            hyps.append(m)
        # (curved triangles only: a flat face has its surface already, and
        # every plane containing a motion's direction agrees with it, which
        # on a printed mount made patches of 20,000 triangles that only cost
        # time, and a stray small one that changed the faces around it)
        flat = np.array([r.kind == "plane" and not r.weak for r in self.regions], dtype=bool)
        faces = faces[~flat[lab[faces]]]
        cen = self.V[F[faces]].mean(1)
        used = np.zeros(len(F), dtype=bool)
        lim = np.sin(np.radians(max_deg))
        out = []
        for mot in hyps[:60]:
            v = mot.velocity(cen)
            sp = np.linalg.norm(v, axis=1)
            ok = (sp > 1e-12) & (np.abs((fn[faces] * v).sum(1)) <= lim * np.maximum(sp, 1e-300)) & ~used[faces]
            if ok.sum() < min_tris:
                continue
            for comp in _cluster_components(faces[ok], nb):
                comp = np.asarray(comp)
                if len(comp) < min_tris or len(np.unique(lab[comp])) < min_regions:
                    continue
                m = fit_motion(self.V[F[comp]].mean(1), fn[comp], fa[comp])
                # confirmed only as the motion proposed (fitted on the
                # agreeing triangles alone, a fillet's came out a helix)
                if m is None or m.kind != mot.kind or abs(m.d @ mot.d) < np.cos(np.radians(2.0)):
                    continue
                used[comp] = True
                out.append((comp, m))
        return out

    def _no_worse(self, surf, faces, slack=1.1, many=5):
        """Whether a swept surface or fillet is worth the regions it would
        replace: it follows their triangles clearly more closely (largest
        gap of corners, edge midpoints and centres), or about as closely
        and stands for many of them. On a printed liner, extrusions 1.4
        tolerances off replaced cylinders 0.9 off and the largest gap grew
        threefold; others, as close as the three cylinders they replaced
        by that measure, still stood off the mesh between those points,
        where the cylinders hadn't."""
        new = float(self._gap(surf, faces).max())
        old = 0.0
        lab = self.label
        regs = [r for r in np.unique(lab[faces]) if r >= 0]
        for r in regs:
            g = faces[lab[faces] == r]
            old = max(old, float(self._gap(self.regions[r].prim, g).max()))
        if new <= old / slack:
            return True
        return len(regs) >= many and new <= slack * old + 0.05 * self.tol

    def _take_in_pieces(self, surf, faces, share=0.1):
        """faces of a swept surface, plus the small neighbouring regions
        that lie on it (every corner, edge midpoint and centre within the
        tolerance). At the fan of long thin triangles where a tooth flank
        meets a top or bottom face, a region reaching across the flank's
        border leaves single triangles of the flank outside it (their
        normals are too noisy to count as smooth), which would become tiny
        planes of their own; left out, they make the flank touch itself at
        the fan's vertex, and the cut that separates it leaves necks
        narrower than the tolerance, so the faces come out invalid (a bevel
        gear). Each connected piece of a neighbour outside the surface is
        judged on its own."""
        lab, nb = self.label, self.nb
        inside = np.zeros(len(self.F), dtype=bool)
        inside[faces] = True
        near = nb[faces]
        near = near[near >= 0]
        near = near[~inside[near]]
        out = [faces]
        for l in np.unique(lab[near]):
            if l < 0 or hasattr(self.regions[l].prim, "motion"):
                continue
            g = np.asarray(self.regions[l].faces)
            g = g[(lab[g] == l) & ~inside[g]]
            if not len(g):
                continue
            for piece in _cluster_components(g, nb):
                piece = np.asarray(piece)
                if len(piece) > share * len(faces) or not inside[nb[piece][nb[piece] >= 0]].any():
                    continue
                if self._gap(surf, piece).max() <= self.tol:
                    out.append(piece)
                    inside[piece] = True
        return np.concatenate(out) if len(out) > 1 else faces

    def _revolved_analytic(self, faces):
        """A cone, cylinder, sphere or torus that every triangle of faces
        lies on, or None."""
        P, N, ok, W = self.region_points(faces, True)
        for cls in (Cone, Cylinder, Sphere, Torus):
            prim = self._robust_fit(cls, P[ok], N[ok], W[ok])
            if prim is None or prim.degenerate(self.ext):
                continue
            if self.accept(prim, self.sense_of(prim, faces), faces).all():
                return prim
        return None

    def _make_freeform(self, smooth_deg=35.0):
        """Clusters of facet-like leftovers become freeform B-spline cells.

        A weak region (one flat facet that no surface claimed) is facet-like
        when it meets another weak region across a smooth edge; isolated weak
        regions bounded by sharp edges are genuine small planar faces."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        regs = self.regions
        lab = self.label
        weak = np.array([r.weak for r in regs], dtype=bool)
        if not weak.any():
            return
        nf = len(self.F)
        rows = np.repeat(np.arange(nf), 3)
        cols = self.nb.reshape(-1)
        m = cols >= 0
        rows, cols = rows[m], cols[m]
        cs = (self.fn[rows] * self.fn[cols]).sum(1)
        smooth = (cs >= np.cos(np.radians(smooth_deg))) | self.unsure[rows] | self.unsure[cols]
        lr, lc = lab[rows], lab[cols]
        m = smooth & (lr != lc) & weak[lr] & weak[lc]
        nr = len(regs)
        G = sparse.coo_matrix((np.ones(m.sum(), dtype=np.int8), (lr[m], lc[m])), shape=(nr, nr))
        ncomp, comp = connected_components(G, directed=False)
        sizes = np.bincount(comp, minlength=ncomp)
        ff = weak & (sizes[comp] >= 2)
        if not ff.any():
            return
        ffaces = np.nonzero(ff[lab])[0]
        keep = [i for i in range(nr) if not ff[i]]
        new_regs = [regs[i] for i in keep]
        new_lab = -np.ones(nf, dtype=np.int64)
        for k, i in enumerate(keep):
            new_lab[regs[i].faces] = k
        cells = []
        for cfaces in _cluster_components(ffaces, self.nb):
            cells.extend(fit_freeform_cells(self, cfaces))
        # A cell that can't follow its own triangles (fit_freeform_cells keeps
        # its best try once a cell is too small to split) gives them back to the
        # flat faces it was made from, which follow them exactly: a crease
        # between two big flat faces (a wall and its chamfer) fits no smooth
        # surface, and the B-spline bulged 0.4 mm off it; the facets of a coarse
        # curved surface it can't follow either are better flat than off the
        # mesh. Within the mesh's own coarseness a cell is right (a true curve
        # stands off coarse facets by that much). A flat face goes back whole, so
        # a cell sharing one with a cell that missed goes back too.
        miss = max(self.tol, self.deflection)
        back = np.zeros(nr, dtype=bool)
        bad = [c.prim.dev > miss for c in cells]
        while True:
            for c, b in zip(cells, bad):
                if b:
                    back[np.unique(lab[c.faces])] = True
            more = [not b and back[lab[c.faces]].any() for c, b in zip(cells, bad)]
            if not any(more):
                break
            bad = [b or m for b, m in zip(bad, more)]
        for i in np.nonzero(back)[0]:
            new_lab[regs[i].faces] = len(new_regs)
            new_regs.append(regs[i])
        kept = [c for c, b in zip(cells, bad) if not b]
        for cell in kept:
            cell.parts = [regs[i] for i in np.unique(lab[cell.faces])]
            new_lab[cell.faces] = len(new_regs)
            new_regs.append(cell)
        if back.any():
            self.report.log(f"{int(back.sum())} flat face(s) kept flat: no smooth surface follows them "
                            "(a crease, or a coarse curve too far off its facets)")
        self.freeform_faces = np.concatenate([c.faces for c in kept]) if kept else np.zeros(0, dtype=np.int64)
        self.regions = new_regs
        self.label = new_lab

    def _rebuild_regions(self, refit=True, merge=True):
        """Split disconnected labels, refit primitives and merge equal
        ones (each unless turned off)."""
        from scipy import sparse
        from scipy.sparse.csgraph import connected_components

        nf = len(self.F)
        lab = self.label
        rows = np.repeat(np.arange(nf), 3)
        cols = self.nb.reshape(-1)
        m = (cols >= 0) & (lab[rows] == lab[np.maximum(cols, 0)])
        G = sparse.coo_matrix((np.ones(m.sum(), dtype=np.int8), (rows[m], cols[m])), shape=(nf, nf))
        ncomp, comp = connected_components(G, directed=False)
        old = self.regions
        new_regions = []
        new_label = np.empty(nf, dtype=np.int64)
        order = np.argsort(comp, kind="stable")
        sc = comp[order]
        starts = np.r_[0, np.nonzero(np.diff(sc))[0] + 1, nf]
        for k in range(len(starts) - 1):
            faces = order[starts[k]:starts[k + 1]]
            r = old[lab[faces[0]]]
            prim, sense = r.prim, r.sense
            if not refit or _locked(prim):
                pass
            elif prim.kind not in ("plane", "freeform") and faces.size >= 3:
                P, N, ok, W = self.region_points(faces, True)
                p2 = prim.refit(P[ok], N[ok], W[ok])
                if p2 is not None and not p2.degenerate(self.ext) and self.accept(p2, sense, faces, 1.5 * self.tol).all():
                    prim = p2
                prim, sense = self.simplify(prim, sense, faces)
            elif prim.kind == "plane":
                P, N, ok, W = self.region_points(faces, True)
                p2 = prim.refit(P, None, W)
                if p2.distance(P).max() <= self.tol:
                    prim = p2
            weak = r.weak and prim.kind == "plane"
            new_label[faces] = len(new_regions)
            new_regions.append(Region(prim, faces, sense, weak))
        self.regions = new_regions
        self.label = new_label
        if merge:
            self._merge_equal()

    def _merge_equal(self):
        regs = self.regions
        lab = self.label
        nb = self.nb
        nbl = np.where(nb >= 0, lab[np.maximum(nb, 0)], -1)
        pairs = set()
        for j in range(3):
            a = lab
            b = nbl[:, j]
            m = (b >= 0) & (a != b)
            for x, y in zip(a[m], b[m]):
                pairs.add((min(x, y), max(x, y)))
        parent = list(range(len(regs)))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for x, y in sorted(pairs):
            rx, ry = regs[x], regs[y]
            if rx.kind != ry.kind or rx.kind == "plane" and (rx.weak or ry.weak):
                continue
            if rx.sense != ry.sense:
                continue
            if rx.prim.same_as(ry.prim, self.tol, self.ext):
                fx, fy = find(x), find(y)
                if fx != fy:
                    parent[fy] = fx
        roots = {}
        out = []
        for i, r in enumerate(regs):
            f = find(i)
            if f not in roots:
                roots[f] = len(out)
                out.append(Region(regs[f].prim, [], regs[f].sense, regs[f].weak))
            elif _locked(r.prim) and not _locked(out[roots[f]].prim):
                # (a surface stl2curves found, locked, outlives an equal fitted one)
                out[roots[f]].prim, out[roots[f]].sense = r.prim, r.sense
            out[roots[f]].faces.append(r.faces)
        newlab = np.empty_like(lab)
        for r in out:
            r.faces = np.concatenate(r.faces)
        for i, r in enumerate(out):
            newlab[r.faces] = i
        self.regions = out
        self.label = newlab


def _cluster_components(faces, nb):
    from .freeform import _components

    return _components(faces, nb)


def fit_freeform_cells(seg, faces, max_rounds=6):
    """Split a face set into height-field disk cells and fit each with a
    B-spline; cells whose fit misses the tolerance are split again."""
    from .freeform import fit_cell, split_cells, _geodesic_split

    V, F = seg.V, seg.F
    out = []
    todo = split_cells(V, F, faces, seg.fn, seg.fa, seg.nb)
    rounds = 0
    cen = None
    while todo:
        cfaces, n = todo.pop()
        q = float(getattr(seg, "detail", 0.7))
        surf = fit_cell(V, F, cfaces, n, seg.tol, detail=q, max_ctrl=int(16 + 30 * q))
        if surf is None:
            continue
        if surf.dev > seg.tol and len(cfaces) > 8 and rounds < 400:
            rounds += 1
            if cen is None:
                cen = V[F].mean(1)
            parts = _geodesic_split(cfaces, cen, seg.nb, 2)
            if len(parts) >= 2:
                for p in parts:
                    todo.extend(split_cells(V, F, p, seg.fn, seg.fa, seg.nb))
                continue
        reg = Region(surf, cfaces, 1.0, False)
        out.append(reg)
    return out


# ----------------------------------------------------------------------------
# B-rep construction with analytic surfaces
# ----------------------------------------------------------------------------


def _dir(v):
    return gp_Dir(float(v[0]), float(v[1]), float(v[2]))


class CADAssembler(BRepAssembler):
    def __init__(self, V, rg, regions, tol, report=NULL):
        surfaces = [_SurfAdapter(r) for r in regions]
        super().__init__(np.array(V, dtype=np.float64, copy=True), rg, surfaces, tol, report)
        self.regions = regions
        self.sfe = ShapeFix_Edge()
        self.frames = {}
        self.seams = {}
        self.surf_params = {}
        self.fixed_closed = set()
        self.unseamed = set()
        self.sphere_mode = {}
        self.pole_caps = {}
        self.tube_seams = {}
        self.ext = float(np.linalg.norm(V.max(0) - V.min(0)))

    # --- edges ----------------------------------------------------------
    def build_chain_edges(self, chain):
        L = self.regions[chain.left]
        R = self.regions[chain.right] if chain.right >= 0 else None
        both_planar = L.kind == "plane" and (R is None or R.kind == "plane")
        if len(chain.verts) == 2:
            return super().build_chain_edges(chain)
        if R is not None and (L.prim is R.prim or getattr(L.prim, "twin", None) is R.prim) and not chain.closed:
            # one surface cut in two (see split_pinched_regions, and the
            # two halves of a swept surface that goes all the way round,
            # SweptSurface.halves): the cut
            # can be any curve on it, so it runs smoothly from corner to
            # corner instead of zigzagging along the mesh's triangles (a
            # zigzag edge folds over itself and makes both faces invalid)
            e = self._on_surface_edge(chain, L.prim)
            if e is not None:
                chain.edges = [e]
                return
        if both_planar:
            if not chain.closed:
                P = self.V[chain.verts]
                a, b = P[0], P[-1]
                d = b - a
                L = np.linalg.norm(d)
                if L > MIN_TOL:
                    d = d / L
                    Q = P - a
                    off = np.linalg.norm(Q - np.outer(Q @ d, d), axis=1).max()
                    if off <= 5 * self.tol:
                        e = self._line_edge(chain.verts[0], chain.verts[-1])
                        self._set_edge_tol(e, chain, off)
                        chain.edges = [e]
                        return
            return super().build_chain_edges(chain)
        if self.chain_is_straight(chain):
            return super().build_chain_edges(chain)
        P = self.V[chain.verts]
        # circle?
        circ = _fit_circle_3d(P[:-1] if chain.closed else P)
        if circ is not None and circ[3] <= self.tol:
            c, n, r, _ = circ
            c, n, r = self._snap_circle(chain, c, n, r)
            e = self._circle_edge(chain, c, n, r)
            if e is not None:
                chain.edges = [e]
                return
        e = self._bspline_edge(chain, P)
        if e is not None:
            chain.edges = [e]
            return
        chain.edges = self._polyline_edges(chain.verts)

    def _on_surface_edge(self, chain, prim, n=24):
        """The straight line between a chain's end corners, laid onto the
        surface, as one smooth edge (None when that doesn't work: the line
        would pass through the surface's axis, say)."""
        a, b = self.V[chain.verts[0]], self.V[chain.verts[-1]]
        if np.linalg.norm(b - a) <= MIN_TOL:
            return None
        t = np.linspace(0.0, 1.0, n)[:, None]
        P = _project(prim, a + t * (b - a))
        if P is None or not np.isfinite(P).all():
            return None
        # a smooth curve: neighbouring samples stay evenly spaced (no jump
        # across the surface where the line ran through its axis)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        if seg.max() > 3.0 * max(seg.mean(), MIN_TOL):
            return None
        P[0], P[-1] = a, b
        curve, dev = fit_chain_curve(P, self.tol)
        if curve is None:
            return None
        S = _sample_curve(curve, 4 * n)
        off = float(prim.distance(S).max())
        if off > self.tol:
            return None
        v0, v1 = self.vertex(chain.verts[0]), self.vertex(chain.verts[-1])
        mk = BRepBuilderAPI_MakeEdge(curve, v0, v1, curve.FirstParameter(), curve.LastParameter())
        if not mk.IsDone():
            return None
        e = mk.Edge()
        self._set_edge_tol(e, chain, max(dev, off))
        return e

    def _set_edge_tol(self, e, chain, curve_dev):
        t = max(MIN_TOL, curve_dev * 1.5 + 1e-9)
        for vi in (chain.verts[0], chain.verts[-1]):
            self.vtol[vi] = max(self.vtol[vi], t)
        self.B.UpdateEdge(e, t)
        for vi in (chain.verts[0], chain.verts[-1]):
            self.B.UpdateVertex(self.vertex(vi), max(MIN_TOL, 1.5 * self.vtol[vi] + 1e-9))

    def _snap_circle(self, chain, c, n, r):
        """A circle around the axis of an adjacent cylinder or cone is one of
        that surface's own circles: take its exact axis and radius, so the
        edge lies exactly on the surface and its start lines up exactly with
        the face's seam (tiny holes and pins otherwise come out slightly
        crossed when other programs read the file)."""
        if not chain.closed:
            return c, n, r  # a short arc's fitted axis is too uncertain
        for rid in (chain.left, chain.right):
            if rid < 0:
                continue
            prim = self.regions[rid].prim
            if prim.kind not in ("cylinder", "cone"):
                continue
            d = prim.d
            if abs(float(n @ d)) < np.cos(np.radians(2.0)):
                continue
            if prim.kind == "cylinder":
                h = float((c - prim.a) @ d)
                c2, r2 = prim.a + h * d, prim.r
            else:
                h = float((c - prim.apex) @ d)
                if h <= 0:
                    continue
                c2, r2 = prim.apex + h * d, h * np.tan(prim.alpha)
            if np.linalg.norm(c2 - c) > self.tol or abs(r2 - r) > self.tol:
                continue
            return c2, (d if float(n @ d) > 0 else -d), float(r2)
        return c, n, r

    def _circle_edge(self, chain, c, n, r):
        P = self.V[chain.verts]
        # orient circle normal so that the chain runs counter-clockwise
        u = unit(P[0] - c - ((P[0] - c) @ n) * n)
        vv = np.cross(n, u)
        ang = np.unwrap(np.arctan2((P - c) @ vv, (P - c) @ u))
        if ang[-1] < ang[0]:
            n = -n
            vv = -vv
            ang = -ang
        ax = gp_Ax2(_pnt(c), _dir(n), _dir(u))
        circ = gp_Circ(ax, float(r))
        from OCP.Geom import Geom_Circle

        gc = Geom_Circle(circ)
        dev = float(np.abs(np.linalg.norm((P - c) - np.outer((P - c) @ n, n), axis=1) - r).max())
        dev = max(dev, float(np.abs((P - c) @ n).max()))
        v0 = self.vertex(chain.verts[0])
        if chain.closed:
            span = ang[-1] - ang[0]
            if abs(span - 2 * np.pi) > 0.3:
                return None
            # vertex position may not be exactly on the circle
            self.vtol[chain.verts[0]] = max(self.vtol[chain.verts[0]], dev)
            self.B.UpdateVertex(v0, max(MIN_TOL, 1.5 * self.vtol[chain.verts[0]] + 1e-9))
            mk = BRepBuilderAPI_MakeEdge(gc, v0, v0, 0.0, 2 * np.pi)
        else:
            v1 = self.vertex(chain.verts[-1])
            a0, a1 = float(ang[0]), float(ang[-1])
            if a1 - a0 > 2 * np.pi - 1e-6:
                return None
            for vi in (chain.verts[0], chain.verts[-1]):
                self.vtol[vi] = max(self.vtol[vi], dev)
                self.B.UpdateVertex(self.vertex(vi), max(MIN_TOL, 1.5 * self.vtol[vi] + 1e-9))
            mk = BRepBuilderAPI_MakeEdge(gc, v0, v1, a0, a1)
        if not mk.IsDone():
            return None
        # an arc through the points can still bulge away between them
        span = (2 * np.pi) if chain.closed else (ang[-1] - ang[0])
        a0 = 0.0 if chain.closed else float(ang[0])
        ts = np.linspace(a0, a0 + span, max(24, 3 * len(P)))
        S = c + r * (np.outer(np.cos(ts), u) + np.outer(np.sin(ts), vv))
        if _polyline_distance(S, P) > self.tol:
            return None
        e = mk.Edge()
        self._set_edge_tol(e, chain, dev)
        return e

    def _bspline_edge(self, chain, P):
        """Smooth curve through the chain points. Candidates are checked
        against the actual polyline (curve samples -> nearest segment) so an
        approximation that overshoots between points is never used."""
        curve, dev = fit_chain_curve(P, self.tol)
        if curve is None:
            return None
        v0 = self.vertex(chain.verts[0])
        v1 = self.vertex(chain.verts[-1])
        ends = [curve.Value(curve.FirstParameter()), curve.Value(curve.LastParameter())]
        for vi, pe in zip((chain.verts[0], chain.verts[-1]), ends):
            d = pe.Distance(_pnt(self.V[vi]))
            self.vtol[vi] = max(self.vtol[vi], d, dev)
            self.B.UpdateVertex(self.vertex(vi), max(MIN_TOL, 1.5 * self.vtol[vi] + 1e-9))
        mk = BRepBuilderAPI_MakeEdge(curve, v0, v1, curve.FirstParameter(), curve.LastParameter())
        if not mk.IsDone():
            return None
        e = mk.Edge()
        self._set_edge_tol(e, chain, dev)
        return e

    # --- seams on periodic surfaces --------------------------------------
    #
    # A face that wraps all the way around a cylinder, cone or torus needs a
    # seam edge: one wire made of [bottom loop, seam, top loop, seam
    # reversed]. Seams are planned before any edge is built so that the seam
    # end points can become real vertices on the (shared) boundary chains.

    def build(self, progress=None):
        self.split_thin_loops()
        self.plan_seams()
        if TURN_SNAP:
            self.snap_turn_cuts()
        return super().build(progress)

    def snap_turn_cuts(self):
        """The corners of a cut between two turns of one surface (split_turns)
        are put exactly on the meridian it was cut at, so the edge between them
        is that meridian's line (_on_surface_edge) and lies on the seam of both
        faces: a face a whole turn wide has no room for its seam anywhere else."""
        for c in self.rg.chains:
            if c.closed or c.left < 0 or c.right < 0:
                continue
            L, R = self.regions[c.left], self.regions[c.right]
            if L.cut is None or L.cut is not R.cut or L.prim is not R.prim:
                continue
            fr = self.region_frame(c.left)
            for vi in (c.verts[0], c.verts[-1]):
                self.V[vi] = fr.at_angle(self.snap_to_surface(c.left, self.V[vi]), 0.0)

    def add_point(self, p):
        self.V = np.vstack([self.V, np.asarray(p, dtype=np.float64)[None]])
        self.vtol = np.r_[self.vtol, 0.0]
        return len(self.V) - 1

    def region_frame(self, r):
        fr = self.frames.get(r)
        if fr is not None:
            return fr
        reg = self.regions[r]
        prim = reg.prim
        P = self.V[np.unique(self.rg.F[reg.faces])]
        if prim.kind == "cylinder":
            o, d = prim.a, prim.d
        elif prim.kind == "cone":
            o, d = prim.apex, prim.d
        elif prim.kind == "torus":
            o, d = prim.c, prim.d
        elif prim.kind == "sphere":
            return self._sphere_frame(r, P)
        else:
            return None
        # (one turn of several: its seam on the meridian it was cut at)
        fr = _Frame(o, d, reg.cut if reg.cut is not None else self._seam_dir(P, o, d))
        self.frames[r] = fr
        return fr

    def _sphere_frame(self, r, P):
        """Small sphere patches get their poles pushed out of the way; big
        caps and bands get the axis of their largest boundary loop so the
        boundary wraps around it (handled with a seam / pole edge)."""
        prim = self.regions[r].prim
        c = prim.c
        dirs = P - c
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1), 1e-30)[:, None]
        m = unit(dirs.mean(0)) if np.any(dirs.mean(0)) else np.array([0, 0, 1.0])
        spread = float(np.degrees(np.arccos(np.clip(dirs @ m, -1, 1)).max())) if len(dirs) else 0.0
        loops = self.rg.region_loops[r]
        if loops and (spread > 70.0 or np.linalg.norm(dirs.mean(0)) < 0.3):
            big = max(loops, key=lambda l: len(self.rg.loop_vertices(l)))
            L = self.V[self.rg.loop_vertices(big)]
            if len(L) >= 3:
                Lc = L.mean(0)
                _, _, vt = np.linalg.svd(L - Lc, full_matrices=False)
                n = vt[2]
                if np.dot(n, m) < 0:
                    n = -n
                fr = _Frame(c, n, self._seam_dir(P, c, n))
                self.sphere_mode[r] = "axial"
                self.frames[r] = fr
                return fr
        u, v = perp_basis(m)
        fr = _Frame(c, u, -m)
        self.sphere_mode[r] = "patch"
        self.frames[r] = fr
        return fr

    def loop_anchors(self, loop):
        out = []
        for cid, fwd in loop:
            c = self.rg.chains[cid]
            if c.closed and cid not in self.fixed_closed:
                continue
            out.append(c.verts[0])
            out.append(c.verts[-1])
        return list(dict.fromkeys(out))

    def plan_seams(self):
        rg = self.rg
        for r, reg in enumerate(self.regions):
            if reg.kind not in PERIODIC and reg.kind != "sphere":
                continue
            loops = rg.region_loops[r]
            if not loops:
                continue
            fr = self.region_frame(r)
            if reg.kind == "sphere" and self.sphere_mode.get(r) != "axial":
                continue
            info = []
            for li, loop in enumerate(loops):
                vs = rg.loop_vertices(loop)
                ang = np.unwrap(fr.angle(self.V[vs + vs[:1]]))
                w = int(np.round((ang[-1] - ang[0]) / (2 * np.pi)))
                info.append((li, w, ang))
            wrap = [x for x in info if x[1] != 0]
            if not wrap:
                if reg.kind == "torus":
                    self._plan_tube_seam(r, loops, fr)
                continue
            # one loop round the axis: a cap round the sphere's pole or the
            # cone's apex (a face with no seam there is invalid)
            if reg.kind in ("sphere", "cone") and len(wrap) == 1:
                self._plan_pole_cap(r, loops, info, wrap[0][0], fr)
                continue
            if len(wrap) != 2:
                self.unseamed.add(r)
                continue
            forb = [(a.min() - 0.03, a.max() + 0.03) for (_, w, a) in info if w == 0]

            def allowed(a):
                for lo, hi in forb:
                    k = np.floor((a - lo) / (2 * np.pi))
                    aa = a - k * 2 * np.pi
                    if lo <= aa <= hi:
                        return False
                return True

            def clearance(a):
                if not forb:
                    return np.pi
                best = np.pi
                for lo, hi in forb:
                    for edge in (lo, hi):
                        dd = abs(_wrap(a - edge))
                        best = min(best, dd)
                return best

            A, Bl = wrap[0][0], wrap[1][0]
            anchors = {A: self.loop_anchors(loops[A]), Bl: self.loop_anchors(loops[Bl])}
            cand = []
            for L in (A, Bl):
                other = Bl if L == A else A
                oang = [fr.angle(self.V[v][None])[0] for v in anchors[other]]
                for v in anchors[L]:
                    a = fr.angle(self.V[v][None])[0]
                    if not allowed(a):
                        continue
                    comp = (not anchors[other]) or any(abs(_wrap(b - a)) < 1e-9 for b in oang)
                    cand.append((comp, clearance(a), a))
            if cand:
                cand.sort(key=lambda t: (not t[0], -t[1]))
                u0 = cand[0][2]
            else:
                vs = rg.loop_vertices(loops[A])
                a = fr.angle(self.V[vs[:1]])[0]
                u0 = a if allowed(a) else _best_gap(forb)
            sv = []
            for L in (A, Bl):
                v = self.ensure_seam_vertex(r, loops[L], u0, fr)
                if v is None:
                    break
                sv.append(v)
            if len(sv) != 2 or sv[0] == sv[1]:
                self.unseamed.add(r)
                continue
            self.frames[r] = fr.rotated(u0)
            self.seams[r] = (A, Bl, sv[0], sv[1])

    def _plan_pole_cap(self, r, loops, info, A, fr):
        forb = [(a.min() - 0.03, a.max() + 0.03) for (_, w, a) in info if w == 0]
        anchors = self.loop_anchors(loops[A])
        u0 = None
        for v in anchors:
            a = fr.angle(self.V[v][None])[0]
            ok = True
            for lo, hi in forb:
                k = np.floor((a - lo) / (2 * np.pi))
                if lo <= a - k * 2 * np.pi <= hi:
                    ok = False
            if ok:
                u0 = a
                break
        if u0 is None:
            if anchors and forb:
                self.unseamed.add(r)
                return
            vs = self.rg.loop_vertices(loops[A])
            u0 = fr.angle(self.V[vs[:1]])[0] if not forb else _best_gap(forb)
        sv = self.ensure_seam_vertex(r, loops[A], u0, fr)
        if sv is None:
            self.unseamed.add(r)
            return
        self.frames[r] = fr.rotated(u0)
        self.pole_caps[r] = (A, sv)

    # --- a seam round a torus's tube ----------------------------------
    #
    # A torus face that goes all the way round its tube but only part way
    # round its axis (a bent rod, a bent pipe's bore: stl2curves hands these
    # over) is bounded by two loops round the tube. Like a face round the axis,
    # it needs a seam: here along one of the torus's parallels (a circle round
    # the axis at one tube angle v0), giving one wire [seam, loop, seam
    # reversed, loop]. Without it the face was invalid and became a few dozen
    # freeform cells (a quarter ring: 24 faces instead of 3).

    def _tube_angle(self, r, P):
        """The angle round the tube (the torus's v) of points P."""
        prim = self.regions[r].prim
        rel = np.atleast_2d(P) - prim.c
        h = rel @ prim.d
        rho = np.linalg.norm(rel - np.outer(h, prim.d), axis=1)
        return np.arctan2(h, rho - prim.R)

    def _tube_turns(self, r, loop):
        vs = self.rg.loop_vertices(loop)
        a = np.unwrap(self._tube_angle(r, self.V[vs + vs[:1]]))
        return int(np.round((a[-1] - a[0]) / (2 * np.pi)))

    def _on_parallel(self, r, p, v0):
        """Point p moved round the tube onto the parallel at tube angle v0."""
        prim = self.regions[r].prim
        fr = self.frames[r]
        u = fr.angle(np.asarray(p, dtype=np.float64)[None])[0]
        radial = np.cos(u) * fr.x + np.sin(u) * fr.y
        return prim.c + (prim.R + prim.r * np.cos(v0)) * radial + prim.r * np.sin(v0) * prim.d

    def _plan_tube_seam(self, r, loops, fr):
        turns = [self._tube_turns(r, loop) for loop in loops]
        wrap = [i for i, w in enumerate(turns) if w != 0]
        if not wrap:
            return
        if len(wrap) != 2:
            self.unseamed.add(r)
            return
        A, Bl = wrap
        # the parallel: through a corner of loop A clear of every other loop
        forb = []
        for i, loop in enumerate(loops):
            if i not in wrap:
                a = np.unwrap(self._tube_angle(r, self.V[self.rg.loop_vertices(loop)]))
                forb.append((a.min() - 0.03, a.max() + 0.03))

        def allowed(a):
            return all(not (lo <= lo + np.mod(a - lo, 2 * np.pi) <= hi) for lo, hi in forb)

        cand = [float(self._tube_angle(r, self.V[v])[0]) for v in self.loop_anchors(loops[A])]
        cand = [a for a in cand if allowed(a)]
        if cand:
            v0 = cand[0]
        elif forb:
            v0 = float(_best_gap(forb))
        else:
            v0 = float(self._tube_angle(r, self.V[self.rg.loop_vertices(loops[A])[:1]])[0])
        sA = self._ensure_tube_seam_vertex(r, loops[A], v0)
        sB = self._ensure_tube_seam_vertex(r, loops[Bl], v0) if sA is not None else None
        if sA is None or sB is None or sA == sB:
            self.unseamed.add(r)
            return
        self.tube_seams[r] = (A, Bl, sA, sB, v0)

    def _ensure_tube_seam_vertex(self, r, loop, v0):
        """A corner on the loop at tube angle v0: an existing one, or a new one
        where the loop crosses that angle (as ensure_seam_vertex round the axis)."""
        rg = self.rg
        for cid, fwd in loop:
            c = rg.chains[cid]
            if c.closed and cid not in self.fixed_closed:
                continue
            for v in (c.verts[0], c.verts[-1]):
                if abs(_wrap(self._tube_angle(r, self.V[v])[0] - v0)) < 1e-9:
                    return v
        for cid, fwd in loop:
            c = rg.chains[cid]
            P = self.V[c.verts]
            rel = _wrap(self._tube_angle(r, P) - v0)
            for i in range(len(P) - 1):
                a0, a1 = rel[i], rel[i + 1]
                if abs(a1 - a0) > np.pi or not ((a0 <= 0.0 < a1) or (a1 <= 0.0 < a0)):
                    continue
                t = a0 / (a0 - a1) if a0 != a1 else 0.0
                q = self.add_point(self._on_parallel(r, P[i] + t * (P[i + 1] - P[i]), v0))
                rg.insert_vertex(cid, i + 1, q)
                if c.closed:
                    rg.rotate_closed_chain(cid, i + 1)
                    self.fixed_closed.add(cid)
                else:
                    rg.split_chain(cid, i + 1)
                return q
        return None

    def _tube_seam_wire(self, r, face, nat, A, Bl, sA, sB, v0):
        """[seam, loop round the tube at the larger u, seam reversed, the other
        loop]: the face spans u from one loop to the other and v from v0 to
        v0 + 2 pi; the seam is used at v0 one way and at v0 + 2 pi the other."""
        from OCP.Geom import Geom_Circle

        prim = self.regions[r].prim
        fr = self.frames[r]
        tol = max(self.tol, MIN_TOL)
        two_pi = 2 * np.pi
        wA, wB = self._tube_turns(r, nat[A]), self._tube_turns(r, nat[Bl])
        if wA < 0 < wB:
            left, right, sl, sr = A, Bl, sA, sB
        elif wB < 0 < wA:
            left, right, sl, sr = Bl, A, sB, sA
        else:
            raise RuntimeError("inconsistent loops round the tube")
        ul = float(fr.angle(self.V[sl][None])[0])
        ur = ul + float(np.mod(fr.angle(self.V[sr][None])[0] - ul, two_pi))
        for vi in (sl, sr):
            d = float(np.linalg.norm(self.V[vi] - self._on_parallel(r, self.V[vi], v0)))
            self.vtol[vi] = max(self.vtol[vi], d)
            self.B.UpdateVertex(self.vertex(vi), max(MIN_TOL, 1.5 * self.vtol[vi] + 1e-9))
        centre = prim.c + prim.r * np.sin(v0) * prim.d
        geom = Geom_Circle(gp_Ax2(_pnt(centre), _dir(prim.d), _dir(fr.x)), float(prim.R + prim.r * np.cos(v0)))
        e = TopoDS_Edge()
        self.B.MakeEdge(e, geom, tol)
        self.B.Add(e, self.vertex(sl))
        self.B.Add(e, self.vertex(sr).Reversed())
        self.B.Range(e, ul, ur)
        pc1 = Geom2d_Line(gp_Pnt2d(0.0, v0), gp_Dir2d(1.0, 0.0))
        pc2 = Geom2d_Line(gp_Pnt2d(0.0, v0 + two_pi), gp_Dir2d(1.0, 0.0))
        self.B.UpdateEdge(e, pc1, pc2, face, tol)
        self.B.Range(e, face, ul, ur)
        self.B.SameParameter(e, True)
        self.B.SameRange(e, True)
        w = TopoDS_Wire()
        self.B.MakeWire(w)
        self.B.Add(w, e)
        for x in self._loop_edges_from(nat[right], sr):
            self.B.Add(w, x)
        self.B.Add(w, e.Reversed())
        for x in self._loop_edges_from(nat[left], sl):
            self.B.Add(w, x)
        w.Closed(True)
        return w

    def ensure_seam_vertex(self, r, loop, u0, fr):
        rg = self.rg
        for cid, fwd in loop:
            c = rg.chains[cid]
            if c.closed and cid not in self.fixed_closed:
                continue
            for v in (c.verts[0], c.verts[-1]):
                if abs(_wrap(fr.angle(self.V[v][None])[0] - u0)) < 1e-9:
                    return v
        ext = self.ext
        for cid, fwd in loop:
            c = rg.chains[cid]
            P = self.V[c.verts]
            rel = _wrap(fr.angle(P) - u0)
            for i in range(len(P) - 1):
                a0, a1 = rel[i], rel[i + 1]
                if abs(a1 - a0) > np.pi:
                    continue  # jump across the opposite side
                if not ((a0 <= 0.0 < a1) or (a1 <= 0.0 < a0) or (a0 < 0.0 <= a1) or (a1 < 0.0 <= a0)):
                    continue
                if abs(a0) < 1e-12 and 0 < i:
                    k = i  # existing interior vertex exactly on the seam
                    q = c.verts[k]
                else:
                    t = a0 / (a0 - a1) if a0 != a1 else 0.0
                    p = P[i] + t * (P[i + 1] - P[i])
                    # a straight-line blend misses the seam angle slightly;
                    # put the point exactly on it (on tiny holes and pins
                    # the miss is enough to make the boundary cross the seam)
                    p = fr.at_angle(self.snap_to_surface(r, p), u0)
                    if np.linalg.norm(p - P[i]) < 1e-9 * ext and i > 0:
                        k, q = i, c.verts[i]
                    elif np.linalg.norm(p - P[i + 1]) < 1e-9 * ext and i + 1 < len(P) - 1:
                        k, q = i + 1, c.verts[i + 1]
                    else:
                        q = self.add_point(p)
                        rg.insert_vertex(cid, i + 1, q)
                        k = i + 1
                if c.closed:
                    rg.rotate_closed_chain(cid, k)
                    self.fixed_closed.add(cid)
                elif 0 < k < len(c.verts) - 1:
                    rg.split_chain(cid, k)
                return q
        return None

    def snap_to_surface(self, r, p):
        prim = self.regions[r].prim
        p = np.asarray(p, dtype=np.float64)
        if prim.kind in ("cylinder", "cone", "torus", "sphere"):
            d = prim.distance(p[None])[0]
            n = prim.normal(p[None])[0]
            if prim.kind == "cone":
                s = prim.signed(p[None])[0]
                return p - s * n
            # move along the surface normal by the signed offset
            if prim.kind == "cylinder":
                rel = p - prim.a
                h = rel @ prim.d
                rad = rel - h * prim.d
                return prim.a + h * prim.d + prim.r * unit(rad)
            if prim.kind == "sphere":
                return prim.c + prim.r * unit(p - prim.c)
            if prim.kind == "torus":
                q = prim._spine(p[None])[0]
                return q + prim.r * unit(p - q)
        return p

    # --- faces ----------------------------------------------------------
    def closed_region_ok(self, r):
        return self.regions[r].kind in ("sphere", "torus")

    def build_face(self, r):
        reg = self.regions[r]
        if reg.kind == "plane":
            return self._planar_face(r, self.surfaces[r], self.rg.region_loops[r])
        if reg.kind == "freeform":
            surf = reg.prim.geom()
        else:
            surf = self._make_surface(reg, r)
        loops = self.rg.region_loops[r]
        if not loops:
            # a closed surface on its own (full sphere or torus)
            from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace

            mf = BRepBuilderAPI_MakeFace(surf, 1e-7)
            if not mf.IsDone():
                raise RuntimeError("closed surface face failed")
            face = mf.Face()
            if reg.sense < 0:
                face.Reverse()
            return face
        tol = max(self.tol, MIN_TOL)
        nat = loops if reg.sense > 0 else [_reverse_loop(l) for l in loops]
        face = self._bounded_face(r, surf, nat, reg.sense < 0, tol)
        if (reg.kind in PERIODIC + ("sphere",) and r not in self.seams and r not in self.pole_caps
                and r not in self.tube_seams):
            # On a closed surface a boundary running the wrong way round
            # trims the complementary part of it: valid, but wrong. The
            # mesh area of the region tells which way is right.
            want = self.region_area(r)
            got = _face_area(face)
            if abs(got - want) > 0.15 * want:
                alt = self._bounded_face(r, surf, [_reverse_loop(l) for l in nat], reg.sense > 0, tol)
                if abs(_face_area(alt) - want) < abs(got - want):
                    face = alt
        return face

    def region_area(self, r):
        if getattr(self, "_fa", None) is None:
            T = self.V[self.rg.F]
            self._fa = 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)
        return float(self._fa[self.regions[r].faces].sum())

    def _bounded_face(self, r, surf, nat, reverse, tol):
        swept = self.regions[r].prim if hasattr(self.regions[r].prim, "pcurve") else None
        face = TopoDS_Face()
        self.B.MakeFace(face, surf, tol)
        wires = []
        seam = self.seams.get(r)
        used = set()
        if seam is not None:
            A, Bl, sA, sB = seam
            wires.append(self._seam_wire(r, face, nat, A, Bl, sA, sB))
            used = {A, Bl}
        elif r in self.pole_caps:
            A, sA = self.pole_caps[r]
            wires.append(self._pole_cap_wire(r, face, nat, A, sA))
            used = {A}
        elif r in self.tube_seams:
            A, Bl, sA, sB, v0 = self.tube_seams[r]
            wires.append(self._tube_seam_wire(r, face, nat, A, Bl, sA, sB, v0))
            used = {A, Bl}
        for i, loop in enumerate(nat):
            if i not in used:
                wires.append(self.wire_for_loop(loop))
        for w in wires:
            exp = TopExp_Explorer(w, TopAbs_EDGE)
            while exp.More():
                e = TopoDS.Edge(exp.Current())
                exp.Next()
                if BRep_Tool.IsClosed_s(e, face):
                    continue  # seam: explicit pcurves already
                if swept is not None:
                    # placed exactly from the sweep (projection onto a
                    # twisted flank went astray: the faces were invalid);
                    # the edge's own curve, fitted along the mesh's border,
                    # can stand off this surface by more than its tolerance,
                    # which then has to cover the gap
                    pc = swept.pcurve(e)
                    if pc is not None:
                        gap = _pcurve_gap(e, pc, swept)
                        self.B.UpdateEdge(e, pc, face, max(BRep_Tool.Tolerance_s(e), 1.5 * gap))
                        continue
                self.sfe.FixAddPCurve(e, face, False, tol)
            sfw = ShapeFix_Wire(w, face, tol)
            sfw.FixShifted()
            self.B.Add(face, sfw.Wire())
        if reverse:
            face.Reverse()
        return face

    def _loop_edges_from(self, loop, start):
        k = None
        for i, (cid, fwd) in enumerate(loop):
            c = self.rg.chains[cid]
            sv = c.verts[0] if fwd else c.verts[-1]
            if sv == start:
                k = i
                break
        if k is None:
            raise RuntimeError("seam vertex not on loop")
        loop = loop[k:] + loop[:k]
        out = []
        for cid, fwd in loop:
            edges = self.rg.chains[cid].edges
            if fwd:
                out.extend(edges)
            else:
                out.extend(e.Reversed() for e in reversed(edges))
        return out

    def _winding(self, r, loop):
        fr = self.frames[r]
        vs = self.rg.loop_vertices(loop)
        ang = np.unwrap(fr.angle(self.V[vs + vs[:1]]))
        return int(np.round((ang[-1] - ang[0]) / (2 * np.pi)))

    def _vparam(self, r, P):
        reg = self.regions[r]
        prim = reg.prim
        sp = self.surf_params[r]
        if prim.kind == "cylinder":
            return float((P - prim.a) @ prim.d)
        if prim.kind == "cone":
            return float(((P - sp["loc"]) @ prim.d) / np.cos(prim.alpha))
        if prim.kind == "torus":
            rel = P - prim.c
            h = rel @ prim.d
            rho = np.linalg.norm(rel - h * prim.d)
            v = np.arctan2(h, rho - prim.R)
            # express in the region's v branch
            v0 = sp["vcut"]
            return float(v0 + np.mod(v - v0, 2 * np.pi))
        if prim.kind == "sphere":
            fr = self.frames[r]
            return float(np.arcsin(np.clip(((P - prim.c) @ fr.d) / prim.r, -1.0, 1.0)))
        raise ValueError(prim.kind)

    def _meridian(self, r, v0, v1):
        from OCP.Geom import Geom_Circle

        prim = self.regions[r].prim
        fr = self.frames[r]
        n = np.cross(fr.x, fr.d)
        geom = Geom_Circle(gp_Ax2(_pnt(prim.c), _dir(n), _dir(fr.x)), prim.r)
        p = lambda v: prim.c + prim.r * (np.cos(v) * fr.x + np.sin(v) * fr.d)
        return geom, p(v0), p(v1)

    def _generator(self, r, v0, v1):
        """A cone's straight line at u=0, parametrised by the surface's own v
        (as _meridian is for a sphere), and its points at v0 and v1."""
        from OCP.Geom import Geom_Line

        prim = self.regions[r].prim
        fr = self.frames[r]
        sp = self.surf_params[r]
        sa, ca = np.sin(prim.alpha), np.cos(prim.alpha)
        g = sa * fr.x + ca * prim.d
        base = sp["loc"] + sp["R0"] * fr.x          # v = 0 on the seam
        return Geom_Line(_pnt(base), _dir(g)), base + v0 * g, base + v1 * g

    def _pole_cap_wire(self, r, face, nat, A, s):
        """Wire of a cap containing a pole: [loop, seam, degenerate pole edge,
        seam reversed]. The pole is a sphere's pole or a cone's apex (a pin's
        point, a drill point, a countersink run to its tip: without the
        degenerate edge such a cone face is invalid, and the InfinityCubeFidget's
        32 cone tips each became a few dozen flat triangles)."""
        prim = self.regions[r].prim
        fr = self.frames[r]
        tol = max(self.tol, MIN_TOL)
        two_pi = 2 * np.pi
        w_a = self._winding(r, nat[A])
        v0 = self._vparam(r, self.V[s])
        north = w_a > 0
        if w_a == 0:
            raise RuntimeError("cap loop does not wrap")
        if prim.kind == "cone":
            # the apex is where the cone's v is lowest (it opens along d), so
            # the face lies below its loop, which runs clockwise in (u, v)
            vp = self._vparam(r, prim.apex)
            if north or not vp < v0:
                raise RuntimeError("cone cap does not hold its apex")
            pole_pt = np.asarray(prim.apex, dtype=np.float64)
        else:
            vp = np.pi / 2 if north else -np.pi / 2
            pole_pt = prim.c + prim.r * np.sign(vp) * fr.d
        pole = TopoDS_Vertex()
        self.B.MakeVertex(pole, _pnt(pole_pt), tol)
        # seam: meridian at u=0 between the loop and the pole, oriented by
        # increasing v
        lo, hi = (v0, vp) if north else (vp, v0)
        if prim.kind == "cone":
            geom, p_lo, p_hi = self._generator(r, lo, hi)
        else:
            geom, p_lo, p_hi = self._meridian(r, lo, hi)
        d = float(np.linalg.norm(self.V[s] - (p_lo if north else p_hi)))
        self.vtol[s] = max(self.vtol[s], d)
        self.B.UpdateVertex(self.vertex(s), max(MIN_TOL, 1.5 * self.vtol[s] + 1e-9))
        seam = TopoDS_Edge()
        self.B.MakeEdge(seam, geom, tol)
        if north:
            self.B.Add(seam, self.vertex(s))
            self.B.Add(seam, pole.Reversed())
        else:
            self.B.Add(seam, pole)
            self.B.Add(seam, self.vertex(s).Reversed())
        self.B.Range(seam, lo, hi)
        pc1 = Geom2d_Line(gp_Pnt2d(two_pi, 0.0), gp_Dir2d(0.0, 1.0))
        pc2 = Geom2d_Line(gp_Pnt2d(0.0, 0.0), gp_Dir2d(0.0, 1.0))
        self.B.UpdateEdge(seam, pc1, pc2, face, tol)
        self.B.Range(seam, face, lo, hi)
        # degenerate edge at the pole
        deg = TopoDS_Edge()
        self.B.MakeEdge(deg)
        self.B.Add(deg, pole)
        self.B.Add(deg, pole.Reversed())
        self.B.Degenerated(deg, True)
        if north:
            pcd = Geom2d_Line(gp_Pnt2d(two_pi, vp), gp_Dir2d(-1.0, 0.0))
        else:
            pcd = Geom2d_Line(gp_Pnt2d(0.0, vp), gp_Dir2d(1.0, 0.0))
        self.B.UpdateEdge(deg, pcd, face, tol)
        self.B.Range(deg, face, 0.0, two_pi)
        self.B.Range(deg, 0.0, two_pi)
        w = TopoDS_Wire()
        self.B.MakeWire(w)
        for x in self._loop_edges_from(nat[A], s):
            self.B.Add(w, x)
        if north:
            self.B.Add(w, seam)
            self.B.Add(w, deg)
            self.B.Add(w, seam.Reversed())
        else:
            self.B.Add(w, seam.Reversed())
            self.B.Add(w, deg)
            self.B.Add(w, seam)
        w.Closed(True)
        return w

    def _seam_wire(self, r, face, nat, A, Bl, sA, sB):
        wA = self._winding(r, nat[A])
        wB = self._winding(r, nat[Bl])
        if wA > 0 > wB:
            bot, top, sbot, stop = A, Bl, sA, sB
        elif wB > 0 > wA:
            bot, top, sbot, stop = Bl, A, sB, sA
        else:
            raise RuntimeError("inconsistent wrapping loops")
        vb = self._vparam(r, self.V[sbot])
        vt = self._vparam(r, self.V[stop])
        if not vt > vb:
            raise RuntimeError("seam loops in unexpected order")
        e = self._seam_edge(r, face, sbot, stop, vb, vt)
        w = TopoDS_Wire()
        self.B.MakeWire(w)
        for x in self._loop_edges_from(nat[bot], sbot):
            self.B.Add(w, x)
        self.B.Add(w, e)
        for x in self._loop_edges_from(nat[top], stop):
            self.B.Add(w, x)
        self.B.Add(w, e.Reversed())
        w.Closed(True)
        return w

    def _seam_edge(self, r, face, sbot, stop, vb, vt):
        from OCP.Geom import Geom_Circle, Geom_Line

        prim = self.regions[r].prim
        fr = self.frames[r]
        sp = self.surf_params[r]
        tol = max(self.tol, MIN_TOL)
        two_pi = 2 * np.pi
        if prim.kind == "cylinder":
            p0 = prim.a + prim.r * fr.x + vb * prim.d
            p1 = prim.a + prim.r * fr.x + vt * prim.d
            geom = Geom_Line(_pnt(p0), _dir(prim.d))
            t0, t1 = 0.0, vt - vb
            pc1 = Geom2d_Line(gp_Pnt2d(two_pi, vb), gp_Dir2d(0.0, 1.0))
            pc2 = Geom2d_Line(gp_Pnt2d(0.0, vb), gp_Dir2d(0.0, 1.0))
        elif prim.kind == "cone":
            loc, R0 = sp["loc"], sp["R0"]
            sa, ca = np.sin(prim.alpha), np.cos(prim.alpha)
            g = sa * fr.x + ca * prim.d
            p0 = loc + (R0 + vb * sa) * fr.x + vb * ca * prim.d
            p1 = loc + (R0 + vt * sa) * fr.x + vt * ca * prim.d
            geom = Geom_Line(_pnt(p0), _dir(g))
            t0, t1 = 0.0, vt - vb
            pc1 = Geom2d_Line(gp_Pnt2d(two_pi, vb), gp_Dir2d(0.0, 1.0))
            pc2 = Geom2d_Line(gp_Pnt2d(0.0, vb), gp_Dir2d(0.0, 1.0))
        elif prim.kind == "sphere":
            geom, p0, p1 = self._meridian(r, vb, vt)
            t0, t1 = vb, vt
            pc1 = Geom2d_Line(gp_Pnt2d(two_pi, 0.0), gp_Dir2d(0.0, 1.0))
            pc2 = Geom2d_Line(gp_Pnt2d(0.0, 0.0), gp_Dir2d(0.0, 1.0))
        elif prim.kind == "torus":
            center = prim.c + prim.R * fr.x
            n = np.cross(fr.x, prim.d)
            geom = Geom_Circle(gp_Ax2(_pnt(center), _dir(n), _dir(fr.x)), prim.r)
            p0 = center + prim.r * (np.cos(vb) * fr.x + np.sin(vb) * prim.d)
            p1 = center + prim.r * (np.cos(vt) * fr.x + np.sin(vt) * prim.d)
            t0, t1 = vb, vt
            pc1 = Geom2d_Line(gp_Pnt2d(two_pi, 0.0), gp_Dir2d(0.0, 1.0))
            pc2 = Geom2d_Line(gp_Pnt2d(0.0, 0.0), gp_Dir2d(0.0, 1.0))
        else:
            raise ValueError(prim.kind)
        for vi, pe in ((sbot, p0), (stop, p1)):
            d = float(np.linalg.norm(self.V[vi] - pe))
            self.vtol[vi] = max(self.vtol[vi], d)
            self.B.UpdateVertex(self.vertex(vi), max(MIN_TOL, 1.5 * self.vtol[vi] + 1e-9))
        e = TopoDS_Edge()
        self.B.MakeEdge(e, geom, tol)
        self.B.Add(e, self.vertex(sbot))
        self.B.Add(e, self.vertex(stop).Reversed())
        self.B.Range(e, t0, t1)
        self.B.UpdateEdge(e, pc1, pc2, face, tol)
        self.B.Range(e, face, t0, t1)
        self.B.SameParameter(e, True)
        self.B.SameRange(e, True)
        return e

    def _make_surface(self, reg, r=None):
        prim = reg.prim
        P = self.V[np.unique(self.F_of(reg))]
        fr = self.region_frame(r) if r is not None else None
        sp = {}
        if prim.kind == "cylinder":
            x = fr.x if fr is not None else self._seam_dir(P, prim.a, prim.d)
            ax3 = gp_Ax3(_pnt(prim.a), _dir(prim.d), _dir(x))
            surf = Geom_CylindricalSurface(ax3, prim.r)
        elif prim.kind == "cone":
            h = (P - prim.apex) @ prim.d
            h0 = float(np.clip(h.mean(), 1e-6, None))
            loc = prim.apex + h0 * prim.d
            x = fr.x if fr is not None else self._seam_dir(P, loc, prim.d)
            ax3 = gp_Ax3(_pnt(loc), _dir(prim.d), _dir(x))
            R0 = h0 * np.tan(prim.alpha)
            surf = Geom_ConicalSurface(ax3, prim.alpha, R0)
            sp = {"loc": loc, "R0": R0}
        elif prim.kind == "sphere":
            if fr is not None:
                ax3 = gp_Ax3(_pnt(prim.c), _dir(fr.d), _dir(fr.x))
            else:
                m = unit((P - prim.c).mean(0)) if len(P) else np.array([0, 0, 1.0])
                if not np.any(m):
                    m = np.array([0, 0, 1.0])
                u, v = perp_basis(m)
                ax3 = gp_Ax3(_pnt(prim.c), _dir(u), _dir(-m))
            surf = Geom_SphericalSurface(ax3, prim.r)
        elif prim.kind == "torus":
            x = fr.x if fr is not None else self._seam_dir(P, prim.c, prim.d)
            ax3 = gp_Ax3(_pnt(prim.c), _dir(prim.d), _dir(x))
            surf = Geom_ToroidalSurface(ax3, prim.R, prim.r)
            # branch cut for the minor angle: middle of the largest gap
            rel = P - prim.c
            hh = rel @ prim.d
            rho = np.linalg.norm(rel - np.outer(hh, prim.d), axis=1)
            vv = np.sort(np.arctan2(hh, rho - prim.R))
            gaps = np.diff(np.r_[vv, vv[0] + 2 * np.pi])
            k = int(np.argmax(gaps))
            sp = {"vcut": float(vv[k] + gaps[k] / 2 - 2 * np.pi)}
        else:
            raise ValueError(prim.kind)
        if r is not None:
            self.surf_params[r] = sp
        return surf

    def F_of(self, reg):
        return self.rg.F[reg.faces]

    @staticmethod
    def _seam_dir(P, origin, d):
        rel = P - origin
        rad = rel - np.outer(rel @ d, d)
        u, v = perp_basis(d)
        ang = np.arctan2(rad @ v, rad @ u)
        # place the seam in the middle of the largest angular gap
        a = np.sort(ang)
        gaps = np.diff(np.r_[a, a[0] + 2 * np.pi])
        k = int(np.argmax(gaps))
        mid = a[k] + gaps[k] / 2
        return np.cos(mid) * u + np.sin(mid) * v


PERIODIC = ("cylinder", "cone", "torus")
# Swept surfaces, tapers and fillets (Segmenter._find_sweeps and
# _find_fillets); MMM_SWEEPS=0 turns them off
import os as _os

FIND_SWEEPS = _os.environ.get("MMM_SWEEPS") != "0"
# Saw-tooth borders straightened (Segmenter._straighten_borders); on by
# default, MMM_STRAIGHTEN=0 turns it off. Owner's rim, mount and tire:
# 154 -> 48, 164 -> 66 and 200 -> 24 saw-tooth edges. The small pieces a cut
# leaves are absorbed again, and a cut that squeezes a surface into a strip
# is undone: one 0.05 mm wide got triangles at the origin in a slicer's
# mesher (OpenCASCADE at 0.003 mm, angle 0.5)
STRAIGHTEN = _os.environ.get("MMM_STRAIGHTEN") != "0"
_PLURAL = {"plane": "planes", "cylinder": "cylinders", "cone": "cones", "sphere": "spheres", "torus": "tori", "freeform": "freeform patches"}


def _pcurve_gap(edge, pc, swept, n=40):
    """How far an edge's 3D curve is from the swept surface's points along
    the edge's curve on it (pc, sharing the edge's parameters)."""
    from OCP.BRepAdaptor import BRepAdaptor_Curve

    c = BRepAdaptor_Curve(edge)
    ts = np.linspace(c.FirstParameter(), c.LastParameter(), n)
    P = np.array([[*c.Value(float(t)).Coord()] for t in ts])
    g = swept.geom()
    Q = np.array([[*g.Value(pc.Value(float(t)).X(), pc.Value(float(t)).Y()).Coord()] for t in ts])
    return float(np.linalg.norm(P - Q, axis=1).max())


def _project(prim, P):
    """Points moved onto a surface (along its normal)."""
    k = prim.kind
    if k == "plane":
        return P - np.outer((P - prim.origin) @ prim.n, prim.n)
    if k == "cylinder":
        rad, h = prim._radial(P)
        rho = np.linalg.norm(rad, axis=1)
        if rho.min() < 1e-6 * max(prim.r, 1.0):
            return None
        return prim.a + np.outer(h, prim.d) + rad / rho[:, None] * prim.r
    if k == "sphere":
        v = P - prim.c
        rho = np.linalg.norm(v, axis=1)
        if rho.min() < 1e-6 * max(prim.r, 1.0):
            return None
        return prim.c + v / rho[:, None] * prim.r
    if k == "torus":
        q = prim._spine(P)
        v = P - q
        rho = np.linalg.norm(v, axis=1)
        if rho.min() < 1e-6 * max(prim.r, 1.0):
            return None
        return q + v / rho[:, None] * prim.r
    if k == "cone":
        Q = P.copy()
        for _ in range(3):
            Q = Q - prim.signed(Q)[:, None] * prim.normal(Q)
        return Q
    if hasattr(prim, "foot"):  # a swept surface
        return prim.X(*prim.foot(P))
    return None


def _face_area(face):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    try:
        g = GProp_GProps()
        BRepGProp.SurfaceProperties_s(face, g, 1e-4)
        return abs(float(g.Mass()))
    except Exception:
        return 0.0


def _wrap(a):
    """Wrap angle(s) to (-pi, pi]."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _best_gap(forb):
    if not forb:
        return 0.0
    edges = sorted((lo % (2 * np.pi), (lo % (2 * np.pi)) + (hi - lo)) for lo, hi in forb)
    best, best_mid = -1.0, 0.0
    for i, (lo, hi) in enumerate(edges):
        nlo = edges[(i + 1) % len(edges)][0] + (2 * np.pi if i + 1 == len(edges) else 0.0)
        gap = nlo - hi
        if gap > best:
            best, best_mid = gap, hi + gap / 2
    return float(_wrap(best_mid))


def _reverse_loop(loop):
    return [(cid, not fwd) for cid, fwd in reversed(loop)]


class _Frame:
    def __init__(self, o, d, x):
        self.o = np.asarray(o, dtype=np.float64)
        self.d = unit(np.asarray(d, dtype=np.float64))
        x = np.asarray(x, dtype=np.float64)
        self.x = unit(x - (x @ self.d) * self.d)
        self.y = np.cross(self.d, self.x)

    def angle(self, P):
        rel = np.atleast_2d(P) - self.o
        return np.arctan2(rel @ self.y, rel @ self.x)

    def rotated(self, a):
        return _Frame(self.o, self.d, np.cos(a) * self.x + np.sin(a) * self.y)

    def at_angle(self, p, a):
        """Rotate point p about the axis so that its angle is exactly a
        (it stays on any surface of revolution about this axis)."""
        rel = np.asarray(p, dtype=np.float64) - self.o
        h = rel @ self.d
        rad = rel - h * self.d
        rho = float(np.linalg.norm(rad))
        return self.o + h * self.d + rho * (np.cos(a) * self.x + np.sin(a) * self.y)


class _SurfAdapter:
    """Gives the base assembler a distance() / normal for any region."""

    def __init__(self, reg):
        self.reg = reg
        self.kind = reg.kind
        if reg.kind == "plane":
            self.origin = reg.prim.origin
            self.normal = reg.prim.n * reg.sense

    def distance(self, P):
        return self.reg.prim.distance(np.asarray(P))


def _polyline_distance(S, P):
    """Max distance from sample points S to the polyline P."""
    A = P[:-1]
    B = P[1:]
    AB = B - A
    L2 = np.maximum((AB * AB).sum(1), 1e-30)
    best = np.full(len(S), np.inf)
    for i0 in range(0, len(A), 256):
        a = A[i0:i0 + 256]
        ab = AB[i0:i0 + 256]
        l2 = L2[i0:i0 + 256]
        t = np.clip(((S[:, None, :] - a[None]) * ab[None]).sum(2) / l2[None], 0, 1)
        proj = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(S[:, None, :] - proj, axis=2).min(1)
        best = np.minimum(best, d)
    return float(best.max())


def _polyline_arc_positions(S, P):
    """Arc-length position along polyline P of each sample's nearest point."""
    A = P[:-1]
    AB = P[1:] - A
    L2 = np.maximum((AB * AB).sum(1), 1e-30)
    seg_len = np.sqrt(L2)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    t = np.clip(((S[:, None, :] - A[None]) * AB[None]).sum(2) / L2[None], 0, 1)
    proj = A[None] + t[..., None] * AB[None]
    d = np.linalg.norm(S[:, None, :] - proj, axis=2)
    i = np.argmin(d, axis=1)
    return cum[i] + t[np.arange(len(S)), i] * seg_len[i], cum[-1]


def _monotonic_along(S, P, slack):
    """True if samples S advance along polyline P without going backwards
    (an overshoot that doubles back along the polyline is invisible to a
    plain distance check but breaks faces on periodic surfaces)."""
    if len(P) < 3 or len(S) < 4:
        return True
    pos, total = _polyline_arc_positions(S[1:-1], P)
    return bool((np.diff(pos) >= -slack).all())


def _sample_curve(curve, n):
    a, b = curve.FirstParameter(), curve.LastParameter()
    out = np.empty((n, 3))
    for i, t in enumerate(np.linspace(a, b, n)):
        p = curve.Value(t)
        out[i] = (p.X(), p.Y(), p.Z())
    return out


def fit_chain_curve(P, tol):
    """Best smooth curve through a chain of points: chord-length cubic
    approximation, then cubic interpolation (periodic for closed chains),
    then an exact linear B-spline. Candidates must stay within ``tol`` of
    the polyline and advance monotonically along it.
    Returns (curve, deviation)."""
    from OCP.Approx import Approx_ChordLength
    from OCP.GeomAPI import GeomAPI_Interpolate
    from OCP.collections import HArray1_gp_Pnt

    P = np.asarray(P, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-9]
    P = P[keep]
    if len(P) < 2:
        return None, 0.0
    closed = len(P) > 3 and np.linalg.norm(P[0] - P[-1]) < 1e-9
    Q = P
    if len(Q) > 400:
        idx = np.unique(np.r_[np.linspace(0, len(Q) - 1, 400).astype(int), len(Q) - 1])
        Q = Q[idx]
    nsamp = max(40, 4 * len(Q))
    slack = 0.25 * tol
    cands = []
    arr = Array1_gp_Pnt(1, len(Q))
    for i, p in enumerate(Q):
        arr.SetValue(i + 1, _pnt(p))
    if closed and len(Q) >= 4:
        try:
            pts = Q[:-1]
            h = HArray1_gp_Pnt(1, len(pts))
            for i, p in enumerate(pts):
                h.SetValue(i + 1, _pnt(p))
            it = GeomAPI_Interpolate(h, True, 1e-9)
            it.Perform()
            if it.IsDone():
                cands.append(it.Curve())
        except Exception:
            pass
    if len(Q) >= 3 and not closed:
        try:
            ap = GeomAPI_PointsToBSpline(arr, Approx_ChordLength, 3, 3, GeomAbs_C2, max(tol * 0.25, 1e-7))
            if ap.IsDone():
                cands.append(ap.Curve())
        except Exception:
            pass
        try:
            h = HArray1_gp_Pnt(1, len(Q))
            for i, p in enumerate(Q):
                h.SetValue(i + 1, _pnt(p))
            it = GeomAPI_Interpolate(h, False, 1e-9)
            it.Perform()
            if it.IsDone():
                cands.append(it.Curve())
        except Exception:
            pass
    for c in cands:
        S = _sample_curve(c, nsamp)
        dev = _polyline_distance(S, P)
        if dev <= tol and _monotonic_along(S, P, slack):
            return c, dev
    # exact fallback: one linear B-spline edge through every point
    from OCP.Geom import Geom_BSplineCurve
    from OCP.collections import Array1_double, Array1_int

    n = len(Q)
    knots = Array1_double(1, n)
    mults = Array1_int(1, n)
    tpar = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]
    for i in range(n):
        knots.SetValue(i + 1, float(tpar[i]))
        mults.SetValue(i + 1, 2 if i in (0, n - 1) else 1)
    try:
        c = Geom_BSplineCurve(arr, knots, mults, 1)
    except Exception:
        return None, 0.0
    return c, (_polyline_distance(Q, P) if len(Q) != len(P) else 0.0)


def _fit_circle_3d(P):
    if len(P) < 3:
        return None
    c = P.mean(0)
    _, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    u, v = vt[0], vt[1]
    Q = np.stack([(P - c) @ u, (P - c) @ v], axis=1)
    from .fitting import fit_circle_2d

    fc = fit_circle_2d(Q)
    if fc is None:
        return None
    c2, r = fc
    center = c + c2[0] * u + c2[1] * v
    rel = P - center
    radial = np.linalg.norm(rel - np.outer(rel @ n, n), axis=1)
    err = max(float(np.abs(radial - r).max()), float(np.abs(rel @ n).max()))
    return center, n, r, err


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def segment_shell(shell, tol, report=NULL):
    seg = Segmenter(shell.V, shell.F, shell.topo, tol, report)
    regions = seg.run()
    return seg, regions


def _self_crossing_regions(asm):
    """Region numbers whose constructed face contains a crossing wire."""
    from OCP.BRepCheck import BRepCheck_Status, BRepCheck_Wire
    from OCP.TopAbs import TopAbs_WIRE
    from OCP.TopExp import TopExp_Explorer

    out = set()
    for r, face in asm.region_faces.items():
        exp = TopExp_Explorer(face, TopAbs_WIRE)
        while exp.More():
            wire = TopoDS.Wire(exp.Current())
            e1, e2 = TopoDS_Edge(), TopoDS_Edge()
            if BRepCheck_Wire(wire).SelfIntersect(face, e1, e2) != BRepCheck_Status.BRepCheck_NoError:
                out.add(r)
                break
            exp.Next()
    return out


def build_cad_shell(shell, tol, report=NULL, max_rounds=6, fit_rounds=3, detail=0.6, seg=None):
    """Segment + build. Analytic faces that cannot be built validly, or that
    don't follow the mesh (checked in the first ``fit_rounds`` rounds), are
    replaced by fitted freeform (B-spline) faces and the shell is rebuilt;
    freeform faces that fail are split into smaller cells. The later rounds
    only clean up faces that are still invalid (often tiny slivers left by
    the splitting, which go to a neighbouring face)."""
    from OCP.BRepCheck import BRepCheck_Analyzer

    from .fidelity import face_blown_up

    if seg is None or abs(seg.tol - tol) > 1e-12 * max(1.0, tol):
        seg = Segmenter(shell.V, shell.F, shell.topo, tol, report)
        seg.detail = detail
        regions = seg.run()
    else:
        regions = seg.regions
    labels = seg.label.copy()
    replaced = 0
    inaccurate = 0
    crossed_this_round = False

    def build(labels, regions):
        labels, regions = split_pinched_regions(seg, labels, regions)
        if TURN_SPLIT:
            labels, regions = split_turns(seg, labels, regions)
        # single triangles cut off from their region (every neighbour in
        # another) make a face OpenCASCADE can't build: they join a
        # neighbour (the ring of a printed mount came out as slivers
        # without this)
        seg.label, seg.regions = labels, regions
        seg._flat_slivers_to_long_edge()
        seg._absorb_isolated_fragments()
        labels, regions = seg.label, seg.regions
        rg = RegionGraph(shell.V, shell.F, labels, shell.topo)
        asm = CADAssembler(shell.V, rg, regions, tol, report)
        faces = asm.build(lambda f: report.progress(0.6 + 0.35 * f))
        _settle_tolerances(asm.shell_from_faces(faces), tol)
        bad = set(r for r, _ in asm.failed_regions)
        for r, f in asm.region_faces.items():
            if not BRepCheck_Analyzer(f).IsValid():
                # an edge can be left with no tolerance at all (a scanned
                # plate's counterbore circle: both its faces were invalid
                # only because of it); settling the face's own tolerances
                # once more fixes that, instead of replacing the face
                _settle_tolerances(f, tol)
            # (a face whose surface blew up passes the validity check)
            if not BRepCheck_Analyzer(f).IsValid() or face_blown_up(f):
                bad.add(r)
        # a face can be valid here and broken in the saved STEP file read
        # back (a STEP file keeps no edge tolerances: an edge saved with 0.1
        # reads back with 1e-7): such faces are built again like broken ones
        # (a straightened border left one on the owner's mount, and the
        # whole mount fell back to smooth surfaces)
        broken = _broken_when_saved(asm) - bad
        if broken:
            report.log(f"{len(broken)} face(s) that would be broken in the saved file are built again")
            bad |= broken
        return labels, regions, asm, faces, bad

    for rnd in range(max_rounds):
        labels, regions, asm, faces, bad = build(labels, regions)
        # A wire crossing is proof that a face boundary is not representable;
        # before turning that whole face into freeform cells, try a local
        # re-labelling of only the crossing face's border strips.  The trial
        # is retained only when rebuilding proves it removed a crossing and
        # did not add invalid faces.
        crossing = _self_crossing_regions(asm)
        if crossing and not crossed_this_round:
            crossed_this_round = True
            old_label, old_regions = seg.label, seg.regions
            seg.label, seg.regions = labels.copy(), list(regions)
            if seg._absorb_self_crossing_strips(crossing):
                trial_labels, trial_regions, trial_asm, trial_faces, trial_bad = build(seg.label.copy(), seg.regions)
                # Accept if crossings decreased and no *new* faces went invalid.
                # Faces already in bad are going to be replaced anyway, so only
                # newly-invalid faces count against the trial.
                new_bad = trial_bad - bad
                if (len(_self_crossing_regions(trial_asm)) < len(crossing)
                        and not new_bad):
                    labels, regions, asm, faces, bad = trial_labels, trial_regions, trial_asm, trial_faces, trial_bad
                    report.log("Removed border strips that caused a self-crossing CAD wire")
                else:
                    seg.label, seg.regions = old_label, old_regions
            else:
                seg.label, seg.regions = old_label, old_regions
        crossed_this_round = False  # reset for next round
        if rnd < min(fit_rounds, max_rounds - 1):
            # valid faces that still don't follow their part of the mesh (a
            # surface that bulges between the mesh's vertices, a face whose
            # edges tear it away from the mesh) are replaced the same way
            off = _inaccurate_regions(seg, labels, asm, tol, skip=bad) - bad
            if off:
                report.log(f"{len(off)} surface(s) don't follow the mesh closely enough and are fitted again")
                inaccurate += len(off)
                bad |= off
        else:
            off = set()
        if not bad or rnd == max_rounds - 1:
            break
        replaced += len(bad)
        # a freeform face that still strays from the mesh becomes flat faces
        # (when its triangles are few, and big enough to be faces of their own)
        # (a swept surface that strays becomes freeform patches like any
        # other surface, not flat facets)
        flat = {r for r in off if r < len(regions) and regions[r].kind == "freeform"
                and not hasattr(regions[r].prim, "motion") and _facets_ok(seg, regions[r].faces)}
        labels, regions = _demote(seg, labels, regions, bad - flat, split_more=rnd > 0, facet=flat)
    # Splitting freeform faces can leave slivers too small to be valid faces;
    # those still broken at the end go to a neighbouring face (no new faces
    # are made, so this can't leave new slivers) and the shell is built once more.
    tiny = {r for r in bad if r < len(regions) and _too_small_for_face(seg, labels, regions[r].faces)}
    if tiny:
        labels2, regions2 = _demote(seg, labels, regions, tiny)
        labels2, regions2, asm2, faces2, bad2 = build(labels2, regions2)
        if len(bad2) < len(bad):
            labels, regions, asm, faces, bad = labels2, regions2, asm2, faces2, bad2
    # Faces still broken after every round become flat faces made of their own
    # triangles, as stl2curves leaves a patch it can't build faceted (merged engine):
    # one 5-triangle freeform cell, the only invalid face of 356, failed Rak N-USB's
    # whole body, which then went to another method (775 faces instead of 356).
    last = {r for r in bad if r < len(regions) and _facets_ok(seg, regions[r].faces)}
    if last:
        labels2, regions2 = _demote(seg, labels, regions, set(), facet=last)
        labels2, regions2, asm2, faces2, bad2 = build(labels2, regions2)
        if len(bad2) < len(bad):
            report.log(f"{len(last)} face(s) that still wouldn't build became flat faces of their own triangles")
            labels, regions, asm, faces, bad = labels2, regions2, asm2, faces2, bad2
    # The rounds check each face written out with the others as a loose compound; the
    # saved file holds them as one shell, whose neighbours share each edge's single curve,
    # and that can break faces the compound check passes (and pass ones it flags: on
    # stl2step's test17 the two disagreed both ways). Checked the shell's way during the
    # rounds, faces that later rounds would have mended anyway were rebuilt (test17: 200
    # faces became 225), so it is done once, here: faces still broken that way become
    # flat faces of their own triangles, which follow the mesh exactly (Rak Uni-USB: two
    # freeform faces broke on read-back and the whole merged result was lost).
    saved = {r for r in _broken_when_saved(asm, shell=True) if r < len(regions)}
    if saved:
        flat = {r for r in saved if _facets_ok(seg, regions[r].faces)}
        labels2, regions2 = _demote(seg, labels, regions, saved - flat, facet=flat)
        labels2, regions2, asm2, faces2, bad2 = build(labels2, regions2)
        if len(_broken_when_saved(asm2, shell=True)) < len(saved) and len(bad2) <= len(bad):
            report.log(f"{len(saved)} face(s) that would be broken in the saved file were built again")
            labels, regions, asm, faces, bad = labels2, regions2, asm2, faces2, bad2
    # A flat face of a region's own triangles can still be broken: its triangles were grouped
    # within FACET_GROUP of its plane, so its corners stand off that plane by up to as much as
    # its edges' tolerance, and the check fails on the tie (Solar Arm Mount: 0.0054 mm against
    # 0.0054, and the whole part was saved faceted). Those regions are built once more with
    # only truly coplanar triangles sharing a face (FACET_FLAT): a lone triangle is exactly flat.
    strict = {r for r, f in asm.region_faces.items()
              if r < len(regions) and getattr(regions[r].prim, "facet", False)
              and not BRepCheck_Analyzer(f).IsValid()}
    if strict:
        labels2, regions2 = _demote(seg, labels, regions, set(), facet=strict, strict=strict)
        labels2, regions2, asm2, faces2, bad2 = build(labels2, regions2)
        if len(bad2) < len(bad):
            report.log(f"{len(strict)} flat face(s) that still wouldn't build were split into truly flat pieces")
            labels, regions, asm, faces, bad = labels2, regions2, asm2, faces2, bad2
    if replaced:
        report.log(f"{replaced} surface(s) rebuilt as freeform (B-spline) faces")
    bad_faces = [f for f in faces if not BRepCheck_Analyzer(f).IsValid()]
    if bad_faces:
        # (the area of each face's own triangles, in the order of faces: the repair must
        # not turn a face inside out, see _heal_faces)
        mesh_area = [float(seg.fa[np.asarray(regions[r].faces)].sum()) if r < len(regions) else float("inf")
                     for r in asm.region_faces]
        faces = _heal_faces(asm, faces, bad_faces, tol, report, mesh_area)
    return asm, faces, regions


def _inaccurate_regions(seg, labels, asm, tol, share=0.05, count=3, far_count=None, skip=()):
    """Regions whose built face strays from the mesh triangles it stands
    for, found by measuring both ways (see fidelity.py): points of the face
    far from the mesh (bulges, flaps, torn edges) and points of the region's
    triangles far from the shape (dents, missing material).

    Distances are in units of the tolerance the result is judged by
    (fidelity.Verdict): ``tol``, or the mesh's own chordal deflection when
    it is coarser (a true cylinder stands off the flat sides of a coarse
    mesh's 16-gon, and that is right). A region is inaccurate when at least
    ``count`` of its points are beyond Verdict.P999_K tolerances (a bulge
    or a flap), or more than ``share`` of them beyond Verdict.P99_K. When the
    shell as a whole would still fail the final check (too many far points
    spread thinly over many faces), the faces holding most of them count
    too, until it would pass: fixing those beats losing the whole CAD
    result, and the rest stay as they are."""
    from .fidelity import Surface, Verdict, mesh_surface, sample, tessellate_faces

    # (faces already known to be broken are replaced anyway; meshing them
    # can fail)
    items = [(r, f) for r, f in asm.region_faces.items() if r not in skip]
    if not items:
        return set()
    rng = np.random.default_rng(0)
    V, F = seg.V, seg.F
    SV, SF, owner = tessellate_faces([f for _, f in items], tol * 0.25)
    if len(SF) == 0:
        return set()
    own_region = np.array([r for r, _ in items], dtype=np.int64)[owner]
    mesh, _, spacing = mesh_surface(V, F)
    # (the tolerance the result is judged by: see fidelity.mesh_kind)
    allow = max(tol, mesh.deflection, mesh.noise)
    shape = Surface(SV, SF, max(spacing, tol * 0.5), rng)
    n = int(np.clip(20 * len(F), 20_000, 150_000))
    # face -> mesh
    Q, qt = sample(SV, SF, n, rng)
    dq, _ = mesh.nearest(Q, exact_above=0.5 * allow)
    # mesh -> shape (by the region the triangle belongs to)
    P, pt = sample(V, F, n, rng)
    dp, _ = shape.nearest(P, exact_above=0.5 * allow)
    near, far = Verdict.P99_K * allow, Verdict.P999_K * allow
    out = set()
    for reg_of, d in ((own_region[qt], dq), (labels[pt], dp)):
        ok = reg_of >= 0
        reg_of, d = reg_of[ok], d[ok]
        if not len(d):
            continue
        cnt = np.bincount(reg_of)
        n_near = np.bincount(reg_of, weights=(d > near).astype(float), minlength=len(cnt))
        n_far = np.bincount(reg_of, weights=(d > far).astype(float), minlength=len(cnt))
        out |= {int(r) for r in np.nonzero((n_far >= (far_count or count)) | (n_near > np.maximum(share * cnt, count)))[0]}
        # the whole shell: at most 1% of the points beyond P99_K, 0.1%
        # beyond P999_K (with a little margin)
        for nk, limit in ((n_near, 0.008 * len(d)), (n_far, 0.0008 * len(d))):
            left = nk.sum() - sum(nk[r] for r in out if r < len(nk))
            for r in np.argsort(-nk):
                if left <= limit or nk[r] <= 0:
                    break
                if int(r) not in out:
                    out.add(int(r))
                    left -= nk[r]
    return out


def _heal_faces(asm, faces, bad_faces, tol, report, mesh_area=None):
    """Repair the few faces that are still invalid with OpenCASCADE's face
    healer. A shared ReShape context propagates any edge it replaces to the
    neighbouring faces, so the shell stays watertight.

    That propagation can turn a valid neighbour inside out: on a periodic
    surface, an edge given back the other way round leaves the neighbour
    with everything but its patch (Bravo's motor shaft joint: a fillet strip
    beside a repaired sliver became the whole torus, 696 mm2, 8.3 mm outside
    the part, and the solid read back inside out). With ``mesh_area`` (the
    area of each face's own triangles, in the order of ``faces``), a face
    that comes out more than HEAL_GROWTH times that area, and HEAL_GROWTH_MM2
    more, is taken as such: the broken faces sharing an edge with it are
    left unrepaired and the rest are repaired again."""
    try:
        from OCP.BRepCheck import BRepCheck_Analyzer
        from OCP.ShapeBuild import ShapeBuild_ReShape
        from OCP.ShapeFix import ShapeFix_Face
        from OCP.TopAbs import TopAbs_FACE

        def repaired(heal):
            sh = asm.shell_from_faces(faces)
            ctx = ShapeBuild_ReShape()
            for f in heal:
                sff = ShapeFix_Face(f)
                sff.SetContext(ctx)
                sff.SetPrecision(max(tol * 0.1, 1e-7))
                sff.SetMaxTolerance(max(tol * 5, 1e-4))
                sff.Perform()
                fixed = sff.Face()
                if not fixed.IsSame(f):
                    ctx.Replace(f, fixed)
            healed = ctx.Apply(sh)
            out = []
            exp = TopExp_Explorer(healed, TopAbs_FACE)
            while exp.More():
                out.append(TopoDS.Face(exp.Current()))
                exp.Next()
            return out

        heal = list(bad_faces)
        for _ in range(HEAL_TRIES):
            out = repaired(heal)
            grown = _grown_faces(faces, out, mesh_area)
            if not grown:
                break
            left = [f for f in heal if not any(_share_edge(f, faces[i]) for i in grown)]
            report.log(f"Repairing {len(heal) - len(left)} face(s) turned {len(grown)} neighbour(s) inside out, "
                       "so those are left as they were")
            if len(left) == len(heal):
                return faces
            heal = left
        else:
            return faces
        still = sum(1 for f in out if not BRepCheck_Analyzer(f).IsValid())
        if len(out) >= len(faces) - len(bad_faces) and still < len(bad_faces):
            report.log(f"Repaired {len(bad_faces) - still} of {len(bad_faces)} remaining face(s)")
            return out
    except Exception as e:
        report.log(f"Automatic face repair was skipped ({e})")
    return faces


def _grown_faces(faces, out, mesh_area):
    """Indices of the faces whose repaired version (same position in ``out``:
    ShapeBuild_ReShape.Apply keeps the shell's order) is far bigger than their
    own triangles (_heal_faces). Nothing is found when the counts differ."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps

    if mesh_area is None or not (len(faces) == len(out) == len(mesh_area)):
        return []
    grown = []
    for i, (f, g) in enumerate(zip(faces, out)):
        if g.IsSame(f):
            continue
        props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(g, props)
        area, own = abs(props.Mass()), mesh_area[i]
        if area > HEAL_GROWTH * own and area > own + HEAL_GROWTH_MM2:
            grown.append(i)
    return grown


def _share_edge(a, b):
    """Whether faces a and b share an edge."""
    edges = _ShapeMap()
    TopExp.MapShapes_s(a, TopAbs_EDGE, edges)
    exp = TopExp_Explorer(b, TopAbs_EDGE)
    while exp.More():
        if edges.Contains(exp.Current()):
            return True
        exp.Next()
    return False


def split_pinched_regions(seg, labels, regions, max_passes=4):
    """A region whose triangles meet at a vertex in two separate fans (a
    hole touching the outer boundary at one point, two holes touching) is
    cut in two, so that no face's boundary touches itself. Such faces are
    valid in memory but STEP readers tend to mangle them."""
    from .freeform import seeded_partition

    F = seg.F
    nf = len(F)
    twin = seg.topo.twin
    k = np.arange(3 * nf)
    f = k // 3
    v = F.reshape(-1)
    h_in = 3 * f + (k % 3 + 2) % 3
    t = twin[h_in]
    g = np.where(t >= 0, t // 3, 0)
    cen = None
    labels = labels.copy()
    regions = list(regions)
    for _ in range(max_passes):
        lf = labels[f]
        # first triangle of each fan around a vertex (rotation order)
        start = (t < 0) | (labels[g] != lf)
        # (decoded with the count at this point: regions are added below)
        nr = len(regions) + 1
        key = v[start].astype(np.int64) * nr + lf[start]
        ks, counts = np.unique(key, return_counts=True)
        pinch = ks[counts >= 2]
        if not len(pinch):
            break
        if cen is None:
            cen = seg.V[F].mean(1)
        changed = False
        done = set()
        for pk in pinch:
            vv, ll = int(pk // nr), int(pk % nr)
            if ll in done:
                continue
            seeds = np.unique(f[start & (v == vv) & (lf == ll)])
            if len(seeds) < 2:
                continue
            done.add(ll)
            r = regions[ll]
            parts = _half_turns(r, seg.V[vv], cen) if r.kind in PERIODIC else None
            if parts is None:
                parts = seeded_partition(r.faces, seeds[:2], cen, seg.nb)
            if len(parts) < 2 or not len(parts[0]) or not len(parts[1]):
                continue
            a = Region(r.prim, parts[0], r.sense, r.weak)
            b = Region(r.prim, parts[1], r.sense, r.weak)
            a.facet = b.facet = r.facet
            regions[ll] = a
            labels[parts[1]] = len(regions)
            regions.append(b)
            changed = True
        if not changed:
            break
    return labels, regions


def _half_turns(r, p, cen):
    """A cylinder, cone or torus face that wraps around its axis and touches
    itself at point p is cut into two halves by the angle around the axis,
    starting at p: each half spans about half a turn, which leaves room for
    the surface's seam (a face that wraps almost all the way round gets its
    seam right at the point where it touches itself, and comes out
    invalid). None when the halves wouldn't each be one piece."""
    prim = r.prim
    o = {"cylinder": "a", "cone": "apex", "torus": "c"}[prim.kind]
    o, d = getattr(prim, o), prim.d
    u, v = perp_basis(d)

    def ang(X):
        rel = X - o
        return np.arctan2(rel @ v, rel @ u)

    t = np.mod(ang(cen[r.faces]) - ang(p[None])[0], 2 * np.pi)
    half = t < np.pi
    parts = [r.faces[half], r.faces[~half]]
    if min(len(parts[0]), len(parts[1])) == 0:
        return None
    return parts


FIT_WEIGHT = 30         # border-straightening cut: a triangle moved to the surface it fits a tolerance worse costs this (an edge: ~100; 100 sent test13 faceted)
SLIVER_HEIGHT = 0.05    # tolerances: a triangle this low over its longest edge has no surface of its own (0.01 missed Solar Panel Bracket's)
TURN_SPLIT = True       # cut surfaces winding over a turn into one face a turn
TURN_SNAP = False       # move a cut's corners onto its meridian (off: on the owner's n-mount it made the cut cone pieces stray from the mesh, 231 faces against 193 without it)
TURN_GAP_DEG = 15.0     # degrees: a region whose triangles leave a gap this wide round its axis is under one turn
TURN_SLACK = 0.3        # radians: a boundary loop running further than a turn plus this winds more than once
TURN_CUTS = 72          # meridians tried for the cut, evenly round the axis
MIN_TURN_DEG = 20.0     # degrees: a piece spanning less is a sliver, joined to the piece beside it


def _axis_of(prim):
    o = {"cylinder": "a", "cone": "apex", "torus": "c"}[prim.kind]
    return np.asarray(getattr(prim, o), dtype=np.float64), np.asarray(prim.d, dtype=np.float64)


def split_turns(seg, labels, regions):
    """A cylinder, cone or torus region that winds more than one turn round its
    axis (the strip of cone between a screw's thread at its tapered tip: 2.5
    turns on the owner's n-mount) has a boundary that overlaps itself in the
    surface's parameters, so its face is invalid. CAD makes one face per turn
    (the n-mount's designer: 360 + 360 + 90 degrees), and so is it cut here:
    at the meridian that crosses the fewest of its triangles, by the turn each
    triangle is on, counted along the region's own triangles (the angle
    unwrapped from one to the next: the n-mount's strip also touches itself
    one turn on, which a cut along the meridian alone leaves joined). Each
    piece keeps the surface; the edge between two pieces is laid on it
    (CADAssembler._on_surface_edge). A sliver of a last turn joins the piece
    beside it. Only regions whose boundary runs more than a turn are cut: a
    band round the axis gets its seam (plan_seams)."""
    cen = None
    cands = []
    for ri, r in enumerate(regions):
        if r.kind not in PERIODIC or len(r.faces) < 3:
            continue
        if cen is None:
            cen = seg.V[seg.F].mean(1)
        o, d = _axis_of(r.prim)
        u, v = perp_basis(d)
        rel = cen[r.faces] - o
        ang = np.arctan2(rel @ v, rel @ u)
        a = np.sort(ang)
        if np.diff(np.r_[a, a[0] + 2 * np.pi]).max() > np.radians(TURN_GAP_DEG):
            continue
        cands.append((ri, o, u, v, ang))
    if not cands:
        return labels, regions
    rg = RegionGraph(seg.V, seg.F, labels, seg.topo)
    labels = labels.copy()
    regions = list(regions)
    for ri, o, u, v, ang in cands:
        multi = False
        for loop in rg.region_loops[ri]:
            vs = rg.loop_vertices(loop)
            P = seg.V[vs + vs[:1]] - o
            w = np.unwrap(np.arctan2(P @ v, P @ u))
            # (how far the loop travels round beyond its net winding: a full ring winds once
            # and travels about once, more only by the width of a notch in its border,
            # which counted as a second turn cut two of test17's plain cylinders in four;
            # a helical strip travels 2.5 turns and comes back to a net 0)
            if np.ptp(w) - abs(w[-1] - w[0]) > 2 * np.pi + TURN_SLACK:
                multi = True
                break
        if not multi:
            continue
        r = regions[ri]
        cut = _turn_pieces(seg, r.faces, o, u, v, ang)
        if cut is None:
            continue
        parts, uc = cut
        x = np.cos(uc) * u + np.sin(uc) * v
        a = Region(r.prim, parts[0], r.sense, r.weak)
        a.facet, a.cut = r.facet, x
        regions[ri] = a
        for p in parts[1:]:
            b = Region(r.prim, p, r.sense, r.weak)
            b.facet, b.cut = r.facet, x
            labels[p] = len(regions)
            regions.append(b)
    return labels, regions


def _turn_pieces(seg, faces, o, u, v, ang):
    """(pieces, cut angle) of a region winding more than a turn, one piece a
    turn (see split_turns), or None when it doesn't come apart."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(faces)
    idx = -np.ones(len(seg.F), dtype=np.int64)
    idx[faces] = np.arange(n)
    nb = seg.nb[faces]
    I = np.repeat(np.arange(n), 3)
    J = np.where(nb.reshape(-1) >= 0, idx[np.maximum(nb.reshape(-1), 0)], -1)
    keep = J > I
    I, J = I[keep], J[keep]
    # the angle unwrapped along a spanning tree of the region's triangles
    adj = [[] for _ in range(n)]
    for i, j in zip(I.tolist(), J.tolist()):
        adj[i].append(j)
        adj[j].append(i)
    th = np.full(n, np.nan)
    for s in range(n):
        if not np.isnan(th[s]):
            continue
        th[s] = ang[s]
        stack = [s]
        while stack:
            i = stack.pop()
            for j in adj[i]:
                if np.isnan(th[j]):
                    th[j] = th[i] + _wrap(ang[j] - ang[i])
                    stack.append(j)
    # the cut: the meridian crossing the fewest triangles, preferring one that
    # leaves no sliver of a turn
    T = seg.V[seg.F[faces]] - o
    ta = np.arctan2(T @ v, T @ u)
    span = np.radians(MIN_TURN_DEG)
    best = None
    for uc in np.arange(TURN_CUTS) * (2 * np.pi / TURN_CUTS):
        rel = _wrap(ta - uc)
        cross = int(((rel.min(1) < 0) & (rel.max(1) > 0) & (np.ptp(rel, 1) < np.pi)).sum())
        turn = np.floor((th - uc) / (2 * np.pi)).astype(np.int64)
        sliver = any(np.ptp(th[turn == k]) < span for k in np.unique(turn))
        key = (sliver, cross)
        if best is None or key < best[0]:
            best = (key, turn, uc)
    _, turn, uc = best
    same = turn[I] == turn[J]
    k, comp = connected_components(coo_matrix((np.ones(int(same.sum())), (I[same], J[same])), shape=(n, n)),
                                   directed=False)
    if k < 2:
        return None
    # a sliver joins the piece it shares most edges with
    alone = set()
    for _ in range(k):
        spans = [np.ptp(th[comp == c]) if (comp == c).any() and c not in alone else np.inf for c in range(k)]
        c = int(np.argmin(spans))
        if spans[c] >= span:
            break
        across = (comp[I] != comp[J]) & ((comp[I] == c) | (comp[J] == c))
        other = np.where(comp[I[across]] == c, comp[J[across]], comp[I[across]])
        if not len(other):
            alone.add(c)        # (a piece of the region touching no other)
            continue
        comp[comp == c] = int(np.bincount(other).argmax())
    pieces = [faces[comp == c] for c in np.unique(comp)]
    return (pieces, float(uc)) if len(pieces) > 1 else None


def _settle_tolerances(shape, tol):
    """Let OpenCASCADE measure how far each edge really is from its faces
    and raise tolerances to match (standard healing step). On some short
    arcs this over-estimates in memory; STEP files carry no tolerances, so
    the written geometry is unaffected (every result is re-read in tests)."""
    try:
        BRepLib.SameParameter_s(shape, max(tol * 0.01, 1e-7), True)
        # an edge it gives up on keeps its flag unset, which alone makes
        # its faces invalid (a printed mount: 3 of its 13 broken faces);
        # ShapeFix settles those, without raising their tolerance
        ex = TopExp_Explorer(shape, TopAbs_EDGE)
        while ex.More():
            e = TopoDS.Edge(ex.Current())
            if not BRep_Tool.SameParameter_s(e):
                ShapeFix_Edge().FixSameParameter(e)
            ex.Next()
        BRepLib.UpdateTolerances_s(shape, False)
    except Exception:
        pass


def _demote(seg, labels, regions, bad, split_more=False, facet=(), strict=()):
    """Replace regions in ``bad`` by freeform B-spline cells. Tiny ones
    (about the size of the tolerance, too small for a valid face of their
    own) hand their triangles to the neighbouring surface that fits them
    best instead. Regions in ``facet`` (freeform cells that still don't
    follow the mesh) become flat faces made from their own triangles, which
    follow it exactly."""
    bad = set(bad) | set(facet)
    facet = set(facet)
    tiny = {r for r in bad if _too_small_for_face(seg, labels, regions[r].faces)}
    keep = [r for r in range(len(regions)) if r not in bad]
    new_regions = [regions[r] for r in keep]
    new_labels = np.full_like(labels, -1)
    for i, r in enumerate(keep):
        new_labels[regions[r].faces] = i
    pending = [int(f) for r in tiny for f in regions[r].faces]
    added = {}
    for _ in range(8):
        left = []
        for f in pending:
            best, bd = -1, np.inf
            for g in seg.nb[f]:
                rr = new_labels[g] if g >= 0 else -1
                if rr < 0 or rr == best:
                    continue
                d = float(new_regions[rr].prim.distance(seg.V[seg.F[f]]).max())
                if d < bd:
                    best, bd = rr, d
            if best >= 0:
                new_labels[f] = best
                added.setdefault(best, []).append(f)
            else:
                left.append(f)
        if len(left) == len(pending):
            break
        pending = left
    for rr, fs in added.items():
        old = new_regions[rr]
        reg = Region(old.prim, np.concatenate([old.faces, np.asarray(fs, dtype=np.int64)]), old.sense, old.weak)
        reg.facet = old.facet
        reg.cut = old.cut
        new_regions[rr] = reg
    rest = [r for r in bad if r not in tiny]
    if pending:
        rest.append(None)
    for r in rest:
        faces = regions[r].faces if r is not None else np.asarray(pending, dtype=np.int64)
        if r in facet:
            for cell in _flat_facets(seg, faces, flat=FACET_FLAT if r in strict else FACET_GROUP):
                new_labels[cell.faces] = len(new_regions)
                new_regions.append(cell)
            continue
        cells = fit_freeform_cells(seg, faces)
        if split_more and len(cells) == 1 and len(faces) > 2:
            from .freeform import _geodesic_split

            cen = seg.V[seg.F].mean(1)
            cells = []
            for p in _geodesic_split(faces, cen, seg.nb, 2):
                cells.extend(fit_freeform_cells(seg, p))
        for cell in cells:
            new_labels[cell.faces] = len(new_regions)
            new_regions.append(cell)
    return new_labels, new_regions


def _facets_ok(seg, faces, max_tris=2000, area_k=25.0):
    """Whether a region can become flat faces from its own triangles: not
    too many of them, and on average big enough that a face of one or two
    triangles isn't a sliver (see _too_small_for_face)."""
    faces = np.asarray(faces)
    return 0 < len(faces) <= max_tris and float(seg.fa[faces].mean()) >= area_k * seg.tol ** 2


def _flat_facets(seg, faces, max_deg=0.5, flat=FACET_GROUP):
    """The triangles of ``faces`` as flat faces: neighbouring triangles that
    lie in one plane (within ``max_deg`` and a tenth of the tolerance)
    share a face. Exactly the mesh, so always a faithful last resort."""
    from .fitting import Plane

    faces = np.asarray(faces, dtype=np.int64)
    member = np.zeros(len(seg.F), dtype=bool)
    member[faces] = True
    group = np.full(len(seg.F), -1, dtype=np.int64)
    cos_max = np.cos(np.radians(max_deg))
    out = []
    for f0 in faces:
        if group[f0] >= 0:
            continue
        gid = len(out)
        group[f0] = gid
        stack, comp = [int(f0)], [int(f0)]
        n0 = seg.fn[f0]
        c0 = seg.V[seg.F[f0]].mean(0)
        while stack:
            f = stack.pop()
            for g in seg.nb[f]:
                if g < 0 or not member[g] or group[g] >= 0:
                    continue
                if seg.fn[g] @ n0 < cos_max:
                    continue
                if np.abs((seg.V[seg.F[g]] - c0) @ n0).max() > flat * seg.tol:
                    continue
                group[g] = gid
                stack.append(int(g))
                comp.append(int(g))
        comp = np.array(comp, dtype=np.int64)
        P = seg.V[np.unique(seg.F[comp])]
        n = (seg.fn[comp] * seg.fa[comp, None]).sum(0)
        cell = Region(Plane(P.mean(0), n if np.linalg.norm(n) > 0 else n0), comp, 1.0, False)
        cell.facet = True
        cell.prim.facet = True   # (made from its own triangles; the surface outlives region copies:
                                 # see build_cad_shell's second try)
        out.append(cell)
    return out


def _broken_when_saved(asm, shell=False):
    """Regions whose face is invalid once the faces are written to a STEP
    file and read back: as a loose compound, or with ``shell`` as one
    shell, as the saved file holds them (neighbours then share each edge's
    single 3D curve; a solid made of the shell reads back the same)."""
    import os
    import tempfile

    from OCP.BRep import BRep_Builder
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopoDS import TopoDS_Compound, TopoDS_Shell

    from . import stepio

    rs = sorted(asm.region_faces)
    B = BRep_Builder()
    comp = TopoDS_Shell() if shell else TopoDS_Compound()
    if shell:
        B.MakeShell(comp)
    else:
        B.MakeCompound(comp)
    for r in rs:
        B.Add(comp, asm.region_faces[r])
    fd, path = tempfile.mkstemp(suffix=".step")
    os.close(fd)
    try:
        stepio.write_step(comp, path)
        back = stepio.read_step(path)
    except Exception:
        return set()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    out = set()
    exp = TopExp_Explorer(back, TopAbs_FACE)
    k = 0
    while exp.More() and k < len(rs):
        if not BRepCheck_Analyzer(exp.Current()).IsValid():
            out.add(rs[k])
        k += 1
        exp.Next()
    # (a different number of faces read back: the order can't be trusted)
    return out if k == len(rs) and not exp.More() else set()


def _too_small_for_face(seg, labels, faces, area_k=25.0, width_k=6.0):
    """Whether a region is too small or too thin to become a valid face of
    its own: about the size of the tolerance, or a sliver whose sides run
    within a few tolerances of each other (their tolerance zones overlap
    and the boundary counts as self-intersecting)."""
    faces = np.asarray(faces)
    area = float(seg.fa[faces].sum())
    tol = seg.tol
    if area <= area_k * tol ** 2:
        return True
    if len(faces) > 64:
        return False
    he = (3 * faces[:, None] + np.arange(3)[None]).reshape(-1)
    twin = seg.topo.twin[he]
    lab = labels[faces[0]]
    outer = (twin < 0) | (labels[np.maximum(twin, 0) // 3] != lab)
    a = seg.V[seg.topo.he_from[he[outer]]]
    b = seg.V[seg.topo.he_to[he[outer]]]
    perim = float(np.linalg.norm(b - a, axis=1).sum())
    return perim > 0 and 2.0 * area / perim <= width_k * tol


def _components(faces, nb):
    faces = list(map(int, faces))
    member = set(faces)
    seen = set()
    out = []
    for f in faces:
        if f in seen:
            continue
        comp = [f]
        seen.add(f)
        stack = [f]
        while stack:
            x = stack.pop()
            for y in nb[x]:
                y = int(y)
                if y >= 0 and y in member and y not in seen:
                    seen.add(y)
                    comp.append(y)
                    stack.append(y)
        out.append(np.array(comp, dtype=np.int64))
    return out


def build_freeform_shell(shell, tol, report=NULL):
    """Every face a fitted B-spline cell (no analytic recognition)."""
    seg = Segmenter(shell.V, shell.F, shell.topo, tol, report)
    regions = fit_freeform_cells(seg, np.arange(len(shell.F)))
    labels = np.empty(len(shell.F), dtype=np.int64)
    for i, r in enumerate(regions):
        labels[r.faces] = i
    rg = RegionGraph(shell.V, shell.F, labels, shell.topo)
    asm = CADAssembler(shell.V, rg, regions, tol, report)
    faces = asm.build()
    sh = asm.shell_from_faces(faces)
    _settle_tolerances(sh, tol)
    return sh


def summarize(regions):
    kinds = {}
    for r in regions:
        kinds[r.kind] = kinds.get(r.kind, 0) + 1
    return kinds


def convert_body_cad(B, body, tol, opts, report=NULL, segs=None):
    from .pipeline import _closed_volume, fix_shape, orient_like_mesh, sew_faces, shells_to_solid
    from . import stepio

    built, flags, all_faces = [], [], []
    failed = False
    kinds = {}
    for shell in body:
        seg = segs.get(id(shell)) if segs else None
        asm, faces, regions = build_cad_shell(shell, tol, report, detail=opts.quality, seg=seg)
        for k, v in summarize(regions).items():
            kinds[k] = kinds.get(k, 0) + v
        if asm.failed_regions:
            failed = True
            report.warn(f"{len(asm.failed_regions)} surface(s) couldn't be built and were left out, so the result may have gaps.")
        all_faces.extend(faces)
        built.append(asm.shell_from_faces(faces))
        flags.append(shell.closed)
    report.log("Found " + ", ".join(f"{v} {_PLURAL.get(k, k) if v != 1 else k}" for k, v in sorted(kinds.items())))
    if failed:
        sewn = sew_faces(all_faces, tol * 2)
        parts = shells_to_solid(sewn)
        shape = parts[0] if len(parts) == 1 else make_compound(parts)
    elif all(flags):
        from .brep import make_solid_checked

        shape = make_solid_checked(B, built, report)
    else:
        shape = make_compound(built)
    mesh_vol = _closed_volume(body)
    shape = orient_like_mesh(shape, mesh_vol)
    if not stepio.is_valid(shape):
        # (the healed shape only when it is valid and keeps the mesh's volume: healing
        # turned the repeater mount's invalid body inside out)
        shape2 = fix_shape(shape, tol, mesh_vol)
        if stepio.is_valid(shape2):
            return shape2
        report.warn("The CAD model still has small errors after automatic repair.")
        return shape
    return shape
