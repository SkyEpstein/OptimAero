"""OptimAero — Aerodynamic Shape Optimizer (plain Tkinter).

Import YOUR shape → the tool deforms its outer surface to cut drag while preserving the inner
volume → export the optimized shape. It refines your geometry; it does not replace it.

Run:  .venv/bin/python -m optimaero.gui_shapeopt
"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from optimaero.shapeopt.optimize import optimize_shape, load_shape, body_aero, make_watertight
from optimaero.shapeopt.envelope import optimize_envelope

STRATEGIES = {"Optimize drone (automatic, CFD)": "auto",
              "Enclose & streamline": "envelope", "Preserve inner volume": "deform",
              "Ducted drone (multirotor)": "ducted", "Airfoil arms (multirotor)": "airfoil"}
OBJECTIVE_LABELS = {"Minimize drag": "min_drag", "Maximize lift": "max_lift", "Maximize L/D": "max_LD"}


class ShapeOptGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.mesh = None
        self.result = None
        root.title("OptimAero — Aerodynamic Shape Optimizer")
        main = ttk.Frame(root, padding=10)
        main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        form = ttk.LabelFrame(main, text="Optimize your shape", padding=10)
        form.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        r = 0
        ttk.Label(form, text="1. Import your shape (CAD)", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w"); r += 1
        ttk.Label(form, text="File units").grid(row=r, column=0, sticky="w")
        self.units = tk.StringVar(value="mm")
        ttk.Combobox(form, textvariable=self.units, values=["mm", "cm", "m", "in"], width=5,
                     state="readonly").grid(row=r, column=1, sticky="w"); r += 1
        ttk.Button(form, text="Import shape…", command=self.on_import).grid(
            row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self.shape_lbl = ttk.Label(form, text="(no shape loaded)", foreground="#777")
        self.shape_lbl.grid(row=r, column=0, columnspan=3, sticky="w"); r += 1

        ttk.Label(form, text="2. Settings", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1
        ttk.Label(form, text="Flow direction").grid(row=r, column=0, sticky="w")
        self.flow_var = tk.StringVar(value="X")
        ttk.Combobox(form, textvariable=self.flow_var, values=["X", "Y", "Z"], width=5,
                     state="readonly").grid(row=r, column=1, sticky="w")
        ttk.Label(form, text="(travel axis)").grid(row=r, column=2, sticky="w"); r += 1
        ttk.Label(form, text="Strategy").grid(row=r, column=0, sticky="w")
        self.strategy = tk.StringVar(value="Optimize drone (automatic, CFD)")
        ttk.Combobox(form, textvariable=self.strategy, values=list(STRATEGIES), width=20,
                     state="readonly").grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        ttk.Label(form, text="Optimize for").grid(row=r, column=0, sticky="w")
        self.objective = tk.StringVar(value="Minimize drag")
        ttk.Combobox(form, textvariable=self.objective, values=list(OBJECTIVE_LABELS), width=20,
                     state="readonly").grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        self.vars = {}
        for label, key, dv, unit in (("Airspeed", "V", 25.0, "m/s"),
                                     ("Angle of attack", "alpha", 0.0, "deg"),
                                     ("Preserve interior", "keep", 0.85, "(0–1)"),
                                     ("Arms (drone mode)", "arms", 4, "#"),
                                     ("Prop diameter", "prop", 0.0, "mm (0=auto)")):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=str(dv)); self.vars[key] = v
            ttk.Entry(form, textvariable=v, width=8).grid(row=r, column=1, sticky="w")
            ttk.Label(form, text=unit).grid(row=r, column=2, sticky="w"); r += 1

        ttk.Label(form, text="Drastic changes").grid(row=r, column=0, sticky="w")
        self.aggr = tk.DoubleVar(value=0.5)
        ttk.Scale(form, from_=0.0, to=1.0, variable=self.aggr, orient="horizontal",
                  length=110).grid(row=r, column=1, sticky="w")
        self.aggr_lbl = ttk.Label(form, text="0.50 (adds fairings outside)")
        self.aggr_lbl.grid(row=r, column=2, sticky="w")
        self.aggr.trace_add("write",
                            lambda *a: self.aggr_lbl.config(text=f"{self.aggr.get():.2f}")); r += 1

        self.run_btn = ttk.Button(form, text="Optimize", command=self.on_run, state="disabled")
        self.run_btn.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 4)); r += 1
        self.save_btn = ttk.Button(form, text="Save optimized shape…", command=self.on_save,
                                   state="disabled")
        self.save_btn.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self.status = ttk.Label(form, text="Import a shape to begin.", foreground="#555")
        self.status.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0)); r += 1
        self.prog = ttk.Progressbar(form, mode="indeterminate", length=220)
        self.prog.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(6, 0)); r += 1

        self.fig = Figure(figsize=(5.6, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d"); self._blank()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)
        self.out = tk.Text(main, height=11, width=90, wrap="word", state="disabled",
                           font=("Menlo", 10))
        self.out.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.root.after(100, self._poll)

    def _blank(self):
        self.ax.clear(); self.ax.set_title("Shape (drag to rotate)")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("z")

    def _plot(self, mesh, title):
        self._blank(); self.ax.set_title(title)
        v = mesh.vertices
        self.ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=mesh.faces,
                             color="#16a34a", alpha=0.6, linewidth=0)
        try:
            self.ax.set_box_aspect((np.ptp(v[:, 0]) or 1, np.ptp(v[:, 1]) or 1, np.ptp(v[:, 2]) or 1))
        except Exception:
            pass
        self.canvas.draw()

    def _plot_pair(self, original, optimized, title):
        self._blank(); self.ax.set_title(title)
        vo = original.vertices
        self.ax.plot_trisurf(vo[:, 0], vo[:, 1], vo[:, 2], triangles=original.faces,
                             color="#9ca3af", alpha=0.35, linewidth=0)   # your shape (gray)
        v = optimized.vertices
        self.ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=optimized.faces,
                             color="#16a34a", alpha=0.45, linewidth=0)    # optimized (green)
        allv = np.vstack([vo, v])
        try:
            self.ax.set_box_aspect((np.ptp(allv[:, 0]) or 1, np.ptp(allv[:, 1]) or 1,
                                    np.ptp(allv[:, 2]) or 1))
        except Exception:
            pass
        self.canvas.draw()

    def on_import(self):
        path = filedialog.askopenfilename(
            title="Import your shape (the part to make aerodynamic)",
            filetypes=[("CAD / mesh", "*.stl *.obj *.ply *.off *.glb *.step *.stp *.iges *.igs"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            raw = load_shape(path, units=self.units.get(), repair=False)
            self.mesh, note = make_watertight(raw)              # solid mesh needed for the additive booleans
            e = self.mesh.extents
            wt = "watertight ✓" if self.mesh.is_watertight else "⚠ NOT watertight"
            self.shape_lbl.config(text=f"loaded: {e[0]:.3f}×{e[1]:.3f}×{e[2]:.3f} m  ({wt})")
            self._plot(self.mesh, "Imported shape")
            self.run_btn.config(state="normal"); self.save_btn.config(state="disabled")
            if note.startswith("repaired"):
                self.status.config(text=f"Loaded & auto-repaired — {note.split(':', 1)[1].strip()}. "
                                        "Set airspeed, then Optimize.")
            elif note.startswith("WARNING"):
                messagebox.showwarning(
                    "Mesh not watertight",
                    "This mesh isn't a closed solid, so the optimizer may not be able to add material "
                    "(it would return your drone unchanged). Re-export it from your CAD tool as a single "
                    "watertight solid (combine/merge the bodies) and re-import.")
                self.status.config(text="Loaded, but NOT watertight — see warning.")
            else:
                self.status.config(text="Shape loaded. Set airspeed, then Optimize.")
        except Exception as e:  # noqa
            messagebox.showerror("Import failed", str(e))

    def on_run(self):
        if self.mesh is None:
            return
        try:
            V = float(self.vars["V"].get()); keep = float(self.vars["keep"].get())
            alpha = float(self.vars["alpha"].get())
            assert V > 0 and 0 < keep < 1
        except Exception:
            messagebox.showerror("Invalid input", "Airspeed > 0 and preserve-interior in (0,1)."); return
        flow = self.flow_var.get().lower(); aggr = float(self.aggr.get())
        strat = STRATEGIES[self.strategy.get()]; obj = OBJECTIVE_LABELS[self.objective.get()]
        try:
            arms = max(2, int(float(self.vars["arms"].get())))
        except Exception:
            arms = 4
        try:
            prop_r = max(0.0, float(self.vars["prop"].get())) / 2000.0   # prop diameter mm → radius m
        except Exception:
            prop_r = 0.0
        if strat == "envelope" and obj in ("max_lift", "max_LD") and abs(alpha) < 1e-6:
            messagebox.showinfo("Set an angle of attack",
                                "Lift is ~0 at 0° angle of attack. Set an angle of attack > 0 to "
                                "optimize for lift or L/D."); return
        self.last_V, self.last_alpha, self.last_flow = V, alpha, flow
        self.run_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        if strat == "auto":
            self.status.config(text="Autonomous CFD optimization — the program is designing the "
                                    "tail + airfoils itself (~4–6 min, 12 CFD candidates)…")
            self.prog.config(mode="determinate", maximum=12, value=0)
        else:
            busy = "Building ducted drone shell…" if strat == "ducted" else "Optimizing your shape…"
            self.status.config(text=busy + " (~30–90 s)")
            self.prog.config(mode="indeterminate"); self.prog.start(12)
        threading.Thread(target=self._work, args=(strat, obj, V, alpha, keep, flow, aggr, arms, prop_r),
                         daemon=True).start()

    @staticmethod
    def _docker_ok() -> bool:
        import shutil
        import subprocess
        if not shutil.which("docker"):
            return False
        try:
            return subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
        except Exception:
            return False

    def _work(self, strat, obj, V, alpha, keep, flow, aggr, arms, prop_r):
        try:
            if strat == "auto":
                from optimaero.drone.segment import segment_multirotor
                from optimaero.drone.optimize import (optimize_drone, optimize_drone_surrogate,
                                                      optimize_drone_general)
                from optimaero.drone.surrogate import surrogate_available
                from optimaero.drone.general_surrogate import general_available
                if not self._docker_ok():
                    raise RuntimeError(
                        "Docker is not running. Automatic drone optimization runs real CFD in Docker "
                        "(the program tries many tail/airfoil designs and keeps the lowest-drag one). "
                        "Start Docker Desktop and retry, or pick 'Enclose & streamline' for a fast "
                        "(non-CFD) estimate.")
                seg = segment_multirotor(self.mesh, up=flow, n_arms=arms, prop_radius=prop_r)
                if not seg.get("rotor_disks"):
                    raise RuntimeError(
                        "This shape doesn't look like a multirotor — no arms/rotor pods were found, so "
                        "there are no arms to airfoil. Check the flow direction and arm count, or use "
                        "'Enclose & streamline' for a general (non-drone) shape.")
                pcb = lambda i, ntot, dr: self.q.put(("prog", (i, ntot, dr)))
                if general_available():
                    # GENERAL surrogate — works on ANY multirotor: score thousands of treatments in ms
                    # (using this drone's shape descriptors), CFD-verify a diverse top-K.
                    ro = optimize_drone_general(self.mesh, seg, V, flow_axis=flow, alpha_deg=alpha,
                                                n_search=8000, top_k=6, progress=pcb)
                elif surrogate_available():
                    ro = optimize_drone_surrogate(self.mesh, seg, V, flow_axis=flow, alpha_deg=alpha,
                                                  n_search=8000, top_k=6, progress=pcb)
                else:
                    ro = optimize_drone(self.mesh, seg, V, flow_axis=flow, n=12, alpha_deg=alpha,
                                        progress=pcb)
                res = {"kind": "auto", "optimized": ro.optimized, "mb": ro.metrics_before,
                       "ma": ro.metrics_after, "ok": ro.contains_original, "params": ro.params,
                       "contains_original": ro.contains_original, "baseline_ok": ro.baseline_ok,
                       "improved": ro.improved, "alpha_used": ro.alpha_deg,
                       "mode": ro.mode, "n_cfd": ro.n_cfd,
                       "surrogate_meta": ro.surrogate_meta}
            elif strat == "envelope":
                r = optimize_envelope(self.mesh, V, flow_axis=flow, objective=obj,
                                      alpha_deg=alpha, aggressiveness=aggr, maxiter=14)
                res = {"kind": "envelope", "optimized": r.optimized, "mb": r.metrics_before,
                       "ma": r.metrics_after, "ok": r.contains_original, "params": r.params}
            elif strat == "ducted":
                from optimaero.drone.segment import segment_multirotor
                from optimaero.drone.ducted import ducted_shell
                seg = segment_multirotor(self.mesh, up=flow, n_arms=arms, prop_radius=prop_r)
                dscale = 1.15 if prop_r > 0 else 2.0     # prop given → duct just clears the prop
                out, shell, mb, ma = ducted_shell(self.mesh, seg, V, flow_axis=flow,
                                                  aggressiveness=aggr, max_len_ratio=2.5,
                                                  duct_scale=dscale)
                res = {"kind": "ducted", "optimized": out, "mb": mb, "ma": ma, "ok": True,
                       "params": {"arms": arms, "rotor_ducts": len(seg["rotor_disks"])}}
            elif strat == "airfoil":
                from optimaero.drone.segment import segment_multirotor
                from optimaero.drone.airfoil import airfoil_arms
                seg = segment_multirotor(self.mesh, up=flow, n_arms=arms, prop_radius=prop_r)
                chord = (0.9 + 1.5 * aggr) * seg["rmax"]   # drastic slider = chord (look ↔ drag: CFD −4% to −19%)
                out = airfoil_arms(self.mesh, seg, flow_axis=flow, chord=chord, thick_scale=0.6)
                mb = body_aero(self.mesh, V, alpha_deg=alpha, flow_axis=flow)
                ma = body_aero(out, V, alpha_deg=alpha, flow_axis=flow)
                res = {"kind": "airfoil", "optimized": out, "mb": mb, "ma": ma, "ok": True,
                       "params": {"arms": arms, "chord_×rmax": round(chord / seg["rmax"], 1)}}
            else:
                r = optimize_shape(self.mesh, V, flow_axis=flow, keepout_frac=keep,
                                   aggressiveness=aggr, maxiter=14)
                mb = body_aero(self.mesh, V, alpha_deg=alpha, flow_axis=flow)
                ma = body_aero(r.optimized, V, alpha_deg=alpha, flow_axis=flow)
                res = {"kind": "deform", "optimized": r.optimized, "mb": mb, "ma": ma,
                       "ok": r.keepout_preserved, "params": r.params}
            self.q.put(("done", res))
        except Exception as e:  # noqa
            self.q.put(("err", str(e)))

    def _poll(self):
        item = None
        try:
            item = self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            if item is not None:
                k, p = item
                if k == "done":
                    self._show(p)
                elif k == "prog":
                    i, ntot, dr = p
                    self.prog.config(maximum=ntot, value=i)
                    extra = f" — last design {dr:.0f} N" if dr < 1e5 else ""
                    self.status.config(text=f"CFD-verifying candidate {i}/{ntot}{extra}…")
                else:                                       # ("err", message)
                    self.prog.stop(); self.run_btn.config(state="normal")
                    self.status.config(text="Error — see dialog.")
                    messagebox.showerror("Optimize failed", p)
        except Exception as e:                              # noqa
            # A display error must NEVER kill the poll loop — otherwise every later run computes a
            # correct result that is silently dropped ("makes no changes"). Surface it, keep polling.
            import traceback
            traceback.print_exc()
            try:
                self.prog.stop(); self.run_btn.config(state="normal")
                self.status.config(text="Display error — see dialog.")
                messagebox.showerror("Display error", f"{type(e).__name__}: {e}")
            except Exception:
                pass
        finally:
            self.root.after(100, self._poll)                # ALWAYS re-arm — the loop must survive

    def _show(self, res):
        self.result = res
        self.prog.stop()
        self.run_btn.config(state="normal")
        title = {"ducted": "Your drone (gray) + ducted shell (green)",
                 "airfoil": "Your drone (gray) + airfoil arms (green)",
                 "auto": "Your drone (gray) + autonomous optimization (green)"}.get(
                     res["kind"], "Your shape (gray) + optimized (green)")
        opt = res.get("optimized")
        if opt is None:
            self._plot(self.mesh, title + " — no valid result")
            self.out.config(state="normal"); self.out.delete("1.0", "end")
            self.out.insert("1.0", "The optimizer did not produce a usable mesh (all CFD candidates "
                                   "failed). Check that Docker is running, then retry.")
            self.out.config(state="disabled")
            self.status.config(text="No valid result — see message."); self.save_btn.config(state="disabled")
            return
        self._plot_pair(self.mesh, opt, title)
        mb, ma = res.get("mb"), res.get("ma")
        V = self.last_V
        is_auto = res["kind"] == "auto"
        alpha = res.get("alpha_used", self.last_alpha) if is_auto else self.last_alpha
        lift_noise = is_auto and abs(alpha) < 1.0     # symmetric drone at ~0° AoA → lift is mesh noise

        # drag line — only claim a % reduction when a real baseline CFD exists
        if mb and ma and mb.get("drag", 0) > 0:
            dred = (1 - ma["drag"] / mb["drag"]) * 100
            dphrase = (f"{dred:.0f}% LESS drag" if dred > 0.5 else
                       f"{-dred:.0f}% MORE drag" if dred < -0.5 else "no drag change")
            ts = (f"   → same-thrust top speed ×{(mb['drag'] / ma['drag']) ** 0.5:.2f}"
                  if 0 < ma["drag"] < mb["drag"] else "")
            drag_line = f"drag:   {mb['drag']:9.3f} N → {ma['drag']:9.3f} N   ({dphrase}){ts}"
        elif ma and ma.get("drag"):
            drag_line = f"drag:   (bare-drone CFD failed — no baseline) → {ma['drag']:9.3f} N"
        else:
            drag_line = "drag:   CFD produced no usable number."

        if is_auto:
            sm = res.get("surrogate_meta") or {}
            if res.get("mode") in ("surrogate", "general") and sm:
                which = "general ML surrogate (any drone)" if res.get("mode") == "general" else "ML surrogate"
                how = (f"{which} scored {sm.get('n_search', 0):,} treatments in ms, then CFD-verified the "
                       f"top {sm.get('top_k_verified', 0)} ({res.get('n_cfd', 0)} CFD runs total)")
            else:
                how = f"CFD-evaluated {res.get('n_cfd', 0)} forms directly"
            if not res.get("improved", True):
                ok = ("no fairing design beat the bare drone in this search — returned your drone "
                      f"UNCHANGED (nothing added, nothing removed). [{how}]")
            elif res.get("contains_original", False):
                ok = (f"autonomous optimization — {how}, kept the lowest-drag one. Additive-only "
                      "VERIFIED (original contained, volume ≥ input); props clear. Real OpenFOAM CFD "
                      "(refine 4 + boundary layers).")
            else:
                ok = ("⚠ additive-only check FAILED on the best design — it may not fully contain your "
                      "drone. Shown for inspection only; do not use as-is.")
        elif res["kind"] == "airfoil":
            ok = ("airfoil arms added — footprint unchanged, props clear (CFD ≈ −4% to −19% by chord; "
                  "the fast estimate below is unreliable for drones — CFD-verify)")
        elif res["kind"] == "ducted":
            ok = (f"ducted drone: {res['params'].get('rotor_ducts', 0)} rotor ducts open, "
                  "drone preserved inside, volume ≥ original")
        elif res["kind"] == "envelope":
            ok = ("encloses your shape: yes" if res["ok"] else
                  "⚠ containment check failed — result may not fully enclose your shape")
        else:
            ok = ("inner volume preserved: yes" if res["ok"] else
                  "⚠ inner volume NOT preserved — shape too concave; lower 'preserve interior'")

        aoa = f"   (at {alpha:.1f}° AoA, {V:.0f} m/s)"
        lines = [drag_line]
        if mb and ma:
            if lift_noise:
                lines.append(f"lift:   ~0 N at {alpha:.0f}° AoA — symmetric body; the CFD value "
                             f"({ma['lift']:+.1f} N) is mesh noise, not real lift.{aoa}")
            else:
                lines += [f"lift:   {mb['lift']:9.3f} N → {ma['lift']:9.3f} N{aoa}",
                          f"L/D:    {mb['LD']:9.2f}   → {ma['LD']:9.2f}"]
        if ma:
            lines.append(f"coeffs: Cd = {ma['Cd']:.4f}"
                         + ("" if lift_noise else f"    Cl = {ma['Cl']:.4f}"))
        lines += [ok, f"params: {{{', '.join(f'{k}={v:.2f}' for k, v in res['params'].items())}}}"]
        lines.append("Real CFD (OpenFOAM); drag is the optimization target." if is_auto
                     else "Fast estimates — CFD-verify before final use. (Lift ~0 at 0° for a symmetric body.)")
        self.out.config(state="normal"); self.out.delete("1.0", "end")
        self.out.insert("1.0", "\n".join(lines)); self.out.config(state="disabled")
        self.status.config(text="Done. Save the optimized shape as CAD.")
        self.save_btn.config(state="normal")

    def on_save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".stl",
                                            filetypes=[("STL", "*.stl"), ("OBJ", "*.obj"),
                                                       ("PLY", "*.ply")])
        if not path:
            return
        try:
            self.result["optimized"].export(path)
            messagebox.showinfo("Saved", f"Wrote {path}")
        except Exception as e:  # noqa
            messagebox.showerror("Export failed", str(e))


def main():
    root = tk.Tk()
    ShapeOptGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
