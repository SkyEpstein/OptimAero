"""OptimAero 3D GUI — aerodynamic-enclosure design (plain Tkinter, no frills).

Enter the component volume your parts must fit inside + an airspeed, press Run, and the app
grows a drag-minimized streamlined enclosure around it, draws its silhouette (with your box
inside), shows the drag in Newtons, and exports a STEP/STL CAD file.

Run:  .venv/bin/python -m optimaero.gui3d
"""
from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cadquery as cq
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from optimaero.three_d.enclosure import Box, optimize_enclosure, _profile, RHO_AIR
from optimaero.three_d import cad3d


class EnclosureGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.result = None
        root.title("OptimAero — Aerodynamic Enclosure Design")
        main = ttk.Frame(root, padding=10); main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        form = ttk.LabelFrame(main, text="Design inputs", padding=10)
        form.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        self.vars = {}

        def field(r, label, key, default, unit=""):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w", pady=2)
            v = tk.StringVar(value=str(default)); self.vars[key] = v
            ttk.Entry(form, textvariable=v, width=10).grid(row=r, column=1, pady=2)
            ttk.Label(form, text=unit).grid(row=r, column=2, sticky="w")

        ttk.Label(form, text="Volume parts must fit inside", font=("", 10, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w")
        field(1, "Length", "lx", 0.30, "m")
        field(2, "Width", "ly", 0.10, "m")
        field(3, "Height", "lz", 0.08, "m")
        ttk.Label(form, text="Operating condition", font=("", 10, "bold")).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        field(5, "Airspeed", "V", 30.0, "m/s")
        ttk.Label(form, text="Purpose").grid(row=6, column=0, sticky="w", pady=(8, 2))
        self.purpose = tk.StringVar(value="Minimum drag")
        ttk.Combobox(form, textvariable=self.purpose, values=["Minimum drag"], width=16,
                     state="readonly").grid(row=6, column=1, columnspan=2, sticky="w")

        self.run_btn = ttk.Button(form, text="Run", command=self.on_run, state="disabled")
        self.run_btn.grid(row=7, column=0, columnspan=3, pady=(12, 4), sticky="ew")
        self.save_btn = ttk.Button(form, text="Save STEP…", command=self.on_save,
                                   state="disabled")
        self.save_btn.grid(row=8, column=0, columnspan=3, sticky="ew")
        self.status = ttk.Label(form, text="Ready. Set inputs and press Run.", foreground="#555")
        self.status.grid(row=9, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.fig = Figure(figsize=(5.4, 3.0), dpi=100)
        self.ax = self.fig.add_subplot(111); self._blank()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)

        self.out = tk.Text(main, height=7, width=84, wrap="word", state="disabled",
                           font=("Menlo", 10))
        self.out.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        self.run_btn.config(state="normal")
        self.root.after(100, self._poll)

    def _blank(self):
        self.ax.clear(); self.ax.set_aspect("equal"); self.ax.grid(alpha=0.3)
        self.ax.set_title("Enclosure silhouette (top view)")
        self.ax.set_xlabel("x (m)"); self.ax.set_ylabel("half-width (m)")

    def on_run(self):
        try:
            box = Box(lx=float(self.vars["lx"].get()), ly=float(self.vars["ly"].get()),
                      lz=float(self.vars["lz"].get()))
            V = float(self.vars["V"].get())
            assert box.lx > 0 and box.ly > 0 and box.lz > 0 and V > 0
        except Exception:
            messagebox.showerror("Invalid input", "Enter positive numbers for all fields.")
            return
        self.run_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        self.status.config(text="Optimizing the enclosure (~10-20s)…")
        threading.Thread(target=self._compute, args=(box, V), daemon=True).start()

    def _compute(self, box, V):
        try:
            r = optimize_enclosure(box, V, maxiter=25)
            self.q.put(("done", (box, V, r)))
        except Exception as e:  # noqa
            self.q.put(("error", str(e)))

    def _poll(self):
        try:
            kind, payload = self.q.get_nowait()
            if kind == "done":
                self._show(*payload)
            elif kind == "error":
                self.run_btn.config(state="normal")
                self.status.config(text="Error — see dialog.")
                messagebox.showerror("Optimization failed", payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _show(self, box, V, r):
        self.result = r
        self.run_btn.config(state="normal")
        # silhouette: body half-width vs x, with the box outline inside
        xi = np.linspace(0, 1, 60)
        w = r.w_max * _profile(xi, r.p)
        x = r.L * xi
        self._blank()
        self.ax.fill_between(x, w, -w, color="#bfe3c6", label="enclosure")
        self.ax.plot(x, w, color="#16a34a"); self.ax.plot(x, -w, color="#16a34a")
        bx = [r.box_x0, r.box_x0 + box.lx]
        self.ax.add_patch(plt_rect(bx[0], -box.ly / 2, box.lx, box.ly))
        self.ax.legend(fontsize=8, loc="upper right"); self.canvas.draw()

        q = 0.5 * RHO_AIR * V ** 2
        frontal = np.pi * r.w_max * r.h_max          # the enclosure's own frontal area
        blunt = q * frontal * 0.9                     # blunt body of the SAME frontal area
        lines = [
            f"Enclosure: length {r.L:.3f} m,  max {2*r.w_max:.3f} × {2*r.h_max:.3f} m "
            f"(W×H),  fineness L/D {r.fineness:.1f}",
            f"Drag at {V:.0f} m/s:  {r.drag:.3f} N   (contains your volume: {r.contains})",
            f"vs a blunt enclosure of the same frontal area (~{blunt:.2f} N):  "
            f"streamlining ~{blunt/r.drag:.0f}× less drag",
            "",
            "Fast-method estimate (inviscid + friction). Stage B adds a CFD-trained surrogate.",
        ]
        self.out.config(state="normal"); self.out.delete("1.0", "end")
        self.out.insert("1.0", "\n".join(lines)); self.out.config(state="disabled")
        self.status.config(text="Done. Save the enclosure as STEP if you like.")
        self.save_btn.config(state="normal")

    def on_save(self):
        if self.result is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".step",
                                            filetypes=[("STEP", "*.step"), ("STL", "*.stl")])
        if not path:
            return
        try:
            solid = cad3d.enclosure_solid(self.result)
            cq.exporters.export(cq.Workplane(obj=solid), path)
            messagebox.showinfo("Saved", f"Wrote {path}")
        except Exception as e:  # noqa
            messagebox.showerror("Export failed", str(e))


def plt_rect(x, y, w, h):
    from matplotlib.patches import Rectangle
    return Rectangle((x, y), w, h, fill=False, edgecolor="#b45309", lw=1.5,
                     label="your volume")


def main():
    root = tk.Tk()
    EnclosureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
