"""Command line interface.

    mmm-engine convert INPUT [-o OUTPUT] [--preset auto|exact|cad|smooth] ...
    mmm-engine inspect INPUT [--preview FILE]
"""
# Part of Manifold Mesh Mix (c) 2026 Tuna-The-First and Claude, AGPL-3.0, commit 4cf72ea.
# Modified 2026-10-26 for the stl2curves merged engine: the error messages are one
# function, failure(), which mms2c/merged.py's command line shares, and they cover
# the path and option checks (pipeline.PathError, pipeline.OptionError).
# Modified 2026-10-26 for Mesh Mix-S2C: Auto and CAD on an STL go through the merged engine
# (mms2c/merged.py: stl2curves' features seed this engine's CAD build, raced against
# both programs alone); --engine mmm keeps this engine's own pipeline.
# Modified 2026-10-31: every mesh format goes through the merged engine for Auto and CAD: 3MF
# projects whole (laid out as on the plate) or split into parts, and OBJ, PLY and OFF files.
import argparse
import json
import os
import sys
import traceback

from .report import Reporter


PREVIEW_TRIS = 1_500_000
TOUCHING = ["ask", "split", "merge"]


def _preview_mesh(V, F):
    """A lighter copy of a very dense mesh for the preview: simplified
    evenly, so the surface stays closed (dropping random triangles would
    leave it full of holes)."""
    import numpy as np

    try:
        import fast_simplification

        V2, F2 = fast_simplification.simplify(
            np.ascontiguousarray(V, dtype=np.float32),
            np.ascontiguousarray(F, dtype=np.int32),
            target_reduction=1.0 - PREVIEW_TRIS / len(F),
        )
        if 0 < len(F2) <= 1.2 * PREVIEW_TRIS:
            return np.asarray(V2, dtype=np.float64), np.asarray(F2, dtype=np.int64)
    except Exception:
        pass
    sel = np.random.default_rng(0).choice(len(F), PREVIEW_TRIS, replace=False)
    return V, F[np.sort(sel)]


def _need_bodies(bodies):
    from .meshio import MeshError

    if not any(len(s.F) for b in bodies for s in b):
        raise MeshError("Nothing is left of this mesh after removing its broken triangles, so there's no shape to use.")


def _fix(args, rep):
    """Check and fix a mesh, then save the fixed copy (STL or 3MF)."""
    from .meshio import load_and_prepare, write_3mf, write_stl

    with rep.section(0.0, 0.95):
        bodies, info = load_and_prepare(args.input, repair=args.repair, remove_debris=not args.keep_debris, report=rep,
                                        layout=args.layout, merge=args.merge_overlaps, touching=args.touching_parts)
    rep.stage("Saving the fixed mesh", 0.95, 1.0)
    _need_bodies(bodies)
    out = args.output
    if out.lower().endswith(".3mf"):
        parts = []
        for p, name in enumerate(info.part_names):
            parts.append((name, [b for b, bp in zip(bodies, info.body_part) if bp == p]))
        write_3mf(out, parts)
    else:
        write_stl(out, bodies)
    rep.result({"output": out, "file_size": os.path.getsize(out), "triangles": info.triangles,
                "bodies": info.bodies, "mesh_check": info.mesh_check})


def _inspect(args, rep):
    import numpy as np

    from . import stepio
    from .meshio import load_and_prepare

    with rep.section(0.0, 0.92):
        bodies, info = load_and_prepare(args.input, repair=args.repair, remove_debris=not args.keep_debris, report=rep,
                                        layout=args.layout, merge=args.merge_overlaps, touching=args.touching_parts)
    rep.stage("Making the 3D preview", 0.92, 1.0)
    _need_bodies(bodies)
    d = info.as_dict()
    d["closed"] = info.open_shells == 0
    d["parts"] = info.parts
    # 3MF projects with several parts: one entry (and preview) per part
    part_list = []
    if info.parts > 1 and len(info.body_part) == len(bodies):
        for p, name in enumerate(info.part_names):
            entry = {"index": p, "name": name, "triangles": info.part_triangles[p], "bbox": info.part_bbox[p]}
            sel = [b for b, bp in zip(bodies, info.body_part) if bp == p]
            if args.part_previews and sel:
                os.makedirs(args.part_previews, exist_ok=True)
                path = os.path.join(args.part_previews, f"part-{p}.mmmp")
                Vs, Fs, off = [], [], 0
                for b in sel:
                    for s in b:
                        Vs.append(s.V)
                        Fs.append(s.F + off)
                        off += len(s.V)
                stepio.export_mesh_preview(np.vstack(Vs), np.vstack(Fs), path)
                entry["preview"] = path
            part_list.append(entry)
    d["part_list"] = part_list
    if args.preview:
        Vs, Fs, off = [], [], 0
        for b in bodies:
            for s in b:
                Vs.append(s.V)
                Fs.append(s.F + off)
                off += len(s.V)
        V = np.vstack(Vs)
        F = np.vstack(Fs)
        if len(F) > PREVIEW_TRIS:
            V, F = _preview_mesh(V, F)
        stepio.export_mesh_preview(V, F, args.preview)
    rep.result(d)


def _merged(args):
    """Whether a conversion goes through the merged engine: Auto and CAD on any mesh (the
    owners' choice, 2026-10-30/31): an STL, OBJ, PLY or OFF file, or a 3MF project whole as
    on its plate or split into parts, one STEP per part. The Smooth and Precise presets, a
    different method, stay this engine's own."""
    return (args.engine == "merged" and args.preset in ("auto", "cad")
            and args.input.lower().endswith((".stl", ".3mf", ".obj", ".ply", ".off")))


def main(argv=None):
    p = argparse.ArgumentParser(prog="mmm-engine", description="Manifold Mesh Mix STL to STEP engine")
    sub = p.add_subparsers(dest="cmd")

    c = sub.add_parser("convert", help="convert a mesh to STEP")
    c.add_argument("input")
    c.add_argument("-o", "--output")
    c.add_argument("--preset", default="auto", choices=["auto", "cad", "smooth", "precise", "exact"])
    c.add_argument("--repair", default="auto", choices=["off", "auto", "aggressive"])
    c.add_argument("--keep-debris", action="store_true")
    c.add_argument("--unit", default="mm", choices=["mm", "cm", "m", "in", "um", "ft"])
    c.add_argument("--tolerance", type=float, default=None, help="absolute fit tolerance (model units)")
    c.add_argument("--quality", "--detail", dest="quality", type=float, default=None,
                   help="0 = fastest, 1 = best quality (default: by the mesh's size for the merged engine, 0.7 otherwise)")
    c.add_argument("--layout", default="parts", choices=["parts", "plate"], help="3MF: one copy of each part, or the full build plate")
    c.add_argument("--max-faces", type=int, default=0)
    c.add_argument("--preview", default=None, help="write a preview file for the app")
    c.add_argument("--no-validate", action="store_true")
    c.add_argument("--split", action="store_true", help="3MF: write one STEP file per part")
    c.add_argument("--part-output", action="append", default=[], help="with --split: output file for part 1, 2, ... (repeat)")
    c.add_argument("--part-preview", action="append", default=[], help="with --split: preview file for part 1, 2, ... (repeat)")
    c.add_argument("--json", action="store_true", help="JSON-lines progress on stdout")
    c.add_argument("--engine", default="merged", choices=["merged", "mmm"],
                   help="merged: stl2curves' features and this engine's CAD build, for Auto and CAD on an STL "
                        "(the default); mmm: this engine's own pipeline")

    c.add_argument("--merge-overlaps", action="store_true", help="join overlapping pieces of a part into one solid")
    c.add_argument("--touching-parts", default="ask", choices=TOUCHING,
                   help="3MF parts that touch each other: keep them separate and report it (ask), keep them "
                        "separate (split), or join them into one part (merge)")

    x = sub.add_parser("fix", help="check and fix a mesh and save the fixed copy (.stl or .3mf)")
    x.add_argument("input")
    x.add_argument("-o", "--output", required=True)
    x.add_argument("--repair", default="auto", choices=["off", "auto", "aggressive"])
    x.add_argument("--keep-debris", action="store_true")
    x.add_argument("--layout", default="parts", choices=["parts", "plate"])
    x.add_argument("--merge-overlaps", action="store_true")
    x.add_argument("--touching-parts", default="ask", choices=TOUCHING)
    x.add_argument("--json", action="store_true")

    i = sub.add_parser("inspect", help="check a mesh and report what was found and fixed")
    i.add_argument("input")
    i.add_argument("--repair", default="auto", choices=["off", "auto", "aggressive"])
    i.add_argument("--keep-debris", action="store_true")
    i.add_argument("--merge-overlaps", action="store_true")
    i.add_argument("--touching-parts", default="ask", choices=TOUCHING)
    i.add_argument("--layout", default="parts", choices=["parts", "plate"])
    i.add_argument("--preview", default=None)
    i.add_argument("--part-previews", default=None, help="folder for one preview per 3MF part")
    i.add_argument("--json", action="store_true")

    sub.add_parser("version")
    sub.add_parser("selfcheck", help="report which optional components are available")

    args = p.parse_args(argv)
    if args.cmd is None:
        p.print_help()
        return 2
    if args.cmd == "version":
        from . import __version__

        print(__version__)
        return 0
    if args.cmd == "selfcheck":
        report = {}
        for mod in ("OCP", "numpy", "scipy", "scipy.interpolate", "trimesh", "pymeshfix", "fast_simplification", "manifold3d", "pynanoinstantmeshes"):
            try:
                m = __import__(mod, fromlist=["_"])
                report[mod] = getattr(m, "__version__", "ok")
            except Exception as e:
                report[mod] = f"MISSING: {e}"
        print(json.dumps(report, indent=1))
        return 0 if all(not str(v).startswith("MISSING") for v in report.values()) else 1

    json_mode = getattr(args, "json", False)
    stream = None
    if json_mode:
        # Keep stdout clean for JSON lines: anything else (including native
        # library chatter) is redirected to stderr at the file descriptor level.
        saved = os.dup(1)
        os.dup2(2, 1)
        stream = os.fdopen(saved, "w", buffering=1)
    rep = Reporter(json_mode=json_mode, stream=stream)
    try:
        if args.cmd == "inspect":
            _inspect(args, rep)
            return 0
        if args.cmd == "fix":
            _fix(args, rep)
            return 0
        from .pipeline import Options, convert

        out = args.output or os.path.splitext(args.input)[0] + ".step"
        opts = Options(
            preset=args.preset,
            repair=args.repair,
            remove_debris=not args.keep_debris,
            unit=args.unit,
            tolerance=args.tolerance,
            quality=args.quality if args.quality is not None else 0.7,
            layout=args.layout,
            max_faces=args.max_faces,
            preview=args.preview,
            validate=not args.no_validate,
            split=args.split,
            part_outputs=args.part_output,
            part_previews=args.part_preview,
            merge_overlaps=args.merge_overlaps,
            touching_parts=args.touching_parts,
        )
        if _merged(args):
            from mms2c import merged

            res = merged.convert_file(args.input, out, args.quality, rep, preset=args.preset, base=opts)
        else:
            res = convert(args.input, out, opts, rep)
        if not rep.json_mode:
            print(json.dumps({k: v for k, v in res.items() if k != "input"}, indent=1))
        return 0
    except Exception as e:  # report and exit non-zero
        return failure(e, args.input, rep)


def failure(e, input_path, rep):
    """Report exception e, raised converting input_path, as a message for the user (a
    JSON error event in JSON mode, with the details), and return the exit code (1)."""
    import errno

    from .meshio import MeshError
    from .pipeline import ConversionError, OptionError, PathError

    reason = None
    if isinstance(e, (MeshError, ConversionError)):
        msg = str(e)
        reason = "The file itself is the problem, not the app."
    elif isinstance(e, PathError):
        msg = str(e)
    elif isinstance(e, OptionError):
        msg = str(e)
        reason = "A setting is out of range."
    elif isinstance(e, MemoryError):
        msg = "Ran out of memory. Move the slider toward Faster, or convert fewer files at once (Advanced → Files at once)."
    elif isinstance(e, FileNotFoundError) and os.path.exists(input_path):
        msg = "The folder to save into couldn't be found. Was it moved or deleted?"
        reason = "The app saves its files there, and it wasn't there any more."
    elif isinstance(e, FileNotFoundError):
        msg = "The file couldn't be found. Was it moved, renamed or deleted?"
    elif isinstance(e, PermissionError):
        system = "macOS" if sys.platform == "darwin" else "Windows" if sys.platform == "win32" else "Linux"
        msg = f"{system} didn't allow reading this file or saving the STEP file in that folder. Choose another output folder in Save STEP files to."
    elif isinstance(e, OSError) and e.errno == errno.ENOSPC:
        msg = "The disk is full, so the STEP file couldn't be saved. Free up some space and convert again."
    elif "STEP write failed" in str(e) or "STEP transfer failed" in str(e):
        msg = "The STEP file couldn't be written. Check that the output folder still exists and has free space."
    else:
        msg = "Something went wrong while converting."
        reason = ("This is a bug in the converter, not a problem with your file. "
                  "Save a report and send it, so it can be fixed.")
    if isinstance(e, MemoryError):
        reason = "Very big or very detailed meshes need a lot of memory, and this one needed more than was free."
    elif isinstance(e, FileNotFoundError) and not os.path.exists(input_path):
        reason = "The app reads the file again when it converts it, and it wasn't there any more."
    elif isinstance(e, PermissionError):
        reason = "The system's file permissions blocked the app."
    details = f"{type(e).__name__}: {e}\n" + traceback.format_exc()
    if rep.json_mode:
        rep.error(msg, reason=reason, details=details)
        sys.stderr.write(traceback.format_exc())
    else:
        print(msg + (f" {reason}" if reason else ""), file=sys.stderr)
        traceback.print_exc()
    return 1


if __name__ == "__main__":
    sys.exit(main())
