"""OptimAero desktop GUI — a plain Tkinter engineering tool (no web, no frills).

Workflow: enter the packaging envelope + operating condition + objective, press Run, and the
app searches CST shape space with the trained surrogate, verifies the optimum with real XFOIL,
draws the airfoil, shows the honest numbers, and exports a STEP file.

Run:  .venv/bin/python -m optimaero.gui
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from optimaero.bakeoff.deploy import TrainedSurrogate
from optimaero.requirements import Envelope, DesignRequirement
from optimaero.optimize.inverse_design import optimize_verified, _score_shape
from optimaero.datasets import uiuc
from optimaero.cad import io as cad

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_XFOIL = os.path.join(_REPO, "tools", "xfoil", "xfoil")
OBJECTIVES = {"Max lift/drag (L/D)": "max_LD", "Max lift (Cl)": "max_Cl",
              "Min drag at target Cl": "min_Cd_at_Cl"}


class OptimAeroGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.result = None
        root.title("OptimAero — Airfoil Inverse Design")

        main = ttk.Frame(root, padding=10)
        main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        # ---- left: inputs ----
        form = ttk.LabelFrame(main, text="Design inputs", padding=10)
        form.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        self.vars = {}

        def row(r, label, key, default, unit=""):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(default)); self.vars[key] = v
            ttk.Entry(form, textvariable=v, width=12).grid(row=r, column=1, pady=2)
            if unit:
                ttk.Label(form, text=unit).grid(row=r, column=2, sticky="w")

        ttk.Label(form, text="Packaging envelope", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w")
        row(1, "Max thickness", "max_t", 0.12, "t/c")
        row(2, "Min thickness", "min_t", 0.08, "t/c")
        ttk.Label(form, text="Operating condition", font=("", 10, "bold")).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        row(4, "Reynolds number", "Re", 1_000_000)
        ttk.Label(form, text="Requirement", font=("", 10, "bold")).grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(form, text="Objective").grid(row=6, column=0, sticky="w", pady=2)
        self.obj = tk.StringVar(value=list(OBJECTIVES)[0])
        ttk.Combobox(form, textvariable=self.obj, values=list(OBJECTIVES), width=20,
                     state="readonly").grid(row=6, column=1, columnspan=2, pady=2, sticky="w")
        row(7, "Target Cl (if used)", "target_cl", 0.8)

        self.run_btn = ttk.Button(form, text="Run", command=self.on_run, state="disabled")
        self.run_btn.grid(row=8, column=0, columnspan=3, pady=(12, 4), sticky="ew")
        self.save_btn = ttk.Button(form, text="Save STEP…", command=self.on_save,
                                   state="disabled")
        self.save_btn.grid(row=9, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(form, text="Loading model…", foreground="#555")
        self.status.grid(row=10, column=0, columnspan=3, sticky="w", pady=(10, 0))

        # ---- right: plot ----
        self.fig = Figure(figsize=(5.2, 3.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self._blank_plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)

        # ---- bottom: results ----
        self.out = tk.Text(main, height=8, width=80, wrap="word", state="disabled",
                           font=("Menlo", 10))
        self.out.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # load the surrogate off the main thread so the window paints immediately
        threading.Thread(target=self._load_model, daemon=True).start()
        self.root.after(100, self._poll)

    # ---------- model / compute ----------
    def _load_model(self):
        surr = TrainedSurrogate()
        self.q.put(("loaded", surr))

    def _compute(self, surr, env, req):
        base, *_ = _score_shape(surr, uiuc.load_coordinates("naca2412"), req)
        best, _ = optimize_verified(surr, req, env, xfoil_path=_XFOIL, n_seeds=3, maxiter=20)
        self.q.put(("done", (base, best)))

    def on_run(self):
        try:
            env = Envelope(max_thickness=float(self.vars["max_t"].get()),
                           min_thickness=float(self.vars["min_t"].get()))
            req = DesignRequirement(Re=float(self.vars["Re"].get()),
                                    objective=OBJECTIVES[self.obj.get()],
                                    target_Cl=float(self.vars["target_cl"].get()))
        except (ValueError, Exception) as e:  # noqa
            messagebox.showerror("Invalid input", str(e)); return
        self.run_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        self.status.config(text="Running: search + XFOIL verify (~30-60s)…")
        threading.Thread(target=self._compute, args=(self.surr, env, req), daemon=True).start()

    def _poll(self):
        try:
            kind, payload = self.q.get_nowait()
            if kind == "loaded":
                self.surr = payload
                self.status.config(text="Ready. Set inputs and press Run.")
                self.run_btn.config(state="normal")
            elif kind == "done":
                self._show(*payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    # ---------- rendering ----------
    def _blank_plot(self):
        self.ax.clear(); self.ax.set_aspect("equal"); self.ax.grid(alpha=0.3)
        self.ax.set_title("Optimized section"); self.ax.set_xlim(-0.05, 1.05)

    def _show(self, base_ld, r):
        self.run_btn.config(state="normal")
        if r is None:
            self.status.config(text="No verified result (XFOIL did not converge). Try again.")
            return
        self.result = r
        self._blank_plot()
        base = uiuc.load_coordinates("naca2412")
        self.ax.plot(base[:, 0], base[:, 1], "--", color="#999", lw=1, label="baseline (naca2412)")
        self.ax.plot(r.coords[:, 0], r.coords[:, 1], "-", color="#16a34a", lw=1.8, label="optimized")
        self.ax.legend(fontsize=8); self.canvas.draw()
        v = f"L/D={r.xfoil_LD:.0f}  Cl={r.xfoil_Cl:.3f}  Cd={r.xfoil_Cd:.4f}" if r.xfoil_LD else "unverified"
        lines = [
            f"XFOIL-VERIFIED optimum:  {v}",
            f"surrogate claimed:       L/D={r.LD:.0f}  (verification is the real number)",
            f"baseline naca2412:       L/D≈{base_ld:.0f}",
            f"thickness t/c={r.thickness:.3f}   envelope satisfied={r.feasible}",
            f"confidence: trusted={r.trusted}  out-of-distribution={r.ood}",
        ]
        self.out.config(state="normal"); self.out.delete("1.0", "end")
        self.out.insert("1.0", "\n".join(lines)); self.out.config(state="disabled")
        self.status.config(text="Done. Save the shape as STEP if you like.")
        self.save_btn.config(state="normal")

    def on_save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".step",
                                            filetypes=[("STEP", "*.step"), ("STL", "*.stl")])
        if not path:
            return
        try:
            if path.lower().endswith(".stl"):
                cad.export_stl(cad.airfoil_to_solid(self.result.coords, chord=0.2), path)
            else:
                cad.export_step(cad.airfoil_to_solid(self.result.coords, chord=0.2), path)
            messagebox.showinfo("Saved", f"Wrote {path}")
        except Exception as e:  # noqa
            messagebox.showerror("Export failed", str(e))


def main():
    root = tk.Tk()
    OptimAeroGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
