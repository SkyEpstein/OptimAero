"""OptimAero — Aircraft Designer GUI (plain Tkinter, no frills).

Pick an aircraft type + mission objective, give the payload volume (type it, or Import a CAD
file of your parts — STEP/IGES/STL/OBJ/PLY/…), press "Design it": the tool DESIGNS an airframe,
tuning real aerodynamic features across hundreds of aero evaluations, draws it in 3D, shows
lift/drag/L-D, and exports it as CAD.

Run:  .venv/bin/python -m optimaero.gui_aircraft
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

from optimaero.aircraft.design import design, DesignSpec, TYPES
from optimaero.aircraft.export import aircraft_mesh, export_aircraft
from optimaero.three_d.enclosure import Box
from optimaero.three_d import cad3d

OBJ_LABEL = {
    "max_LD": "Max lift/drag (efficient / long-range)",
    "max_lift": "Max lift (max payload)",
    "lift_target_min_drag": "Lift a target weight, min drag",
    "min_drag": "Min drag (low-lift / multirotor)",
}
LABEL_OBJ = {v: k for k, v in OBJ_LABEL.items()}
TYPE_LABEL = {"airplane": "Airplane (fixed-wing)", "quadcopter": "Quadcopter"}
LABEL_TYPE = {v: k for k, v in TYPE_LABEL.items()}


class AircraftGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.q: queue.Queue = queue.Queue()
        self.design = None
        root.title("OptimAero — Aircraft Designer")
        main = ttk.Frame(root, padding=10)
        main.grid(sticky="nsew")
        root.columnconfigure(0, weight=1); root.rowconfigure(0, weight=1)

        form = ttk.LabelFrame(main, text="Design inputs", padding=10)
        form.grid(row=0, column=0, sticky="nw", padx=(0, 10))
        self.vars = {}
        r = 0

        ttk.Label(form, text="Aircraft type").grid(row=r, column=0, sticky="w", pady=2)
        self.type_var = tk.StringVar(value=TYPE_LABEL["airplane"])
        tcb = ttk.Combobox(form, textvariable=self.type_var, values=list(TYPE_LABEL.values()),
                           width=24, state="readonly")
        tcb.grid(row=r, column=1, columnspan=2, sticky="w")
        tcb.bind("<<ComboboxSelected>>", self._on_type); r += 1

        ttk.Label(form, text="Objective").grid(row=r, column=0, sticky="w", pady=2)
        self.obj_var = tk.StringVar()
        self.obj_cb = ttk.Combobox(form, textvariable=self.obj_var, width=30, state="readonly")
        self.obj_cb.grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        self._on_type()

        ttk.Label(form, text="Payload volume (parts must fit inside)", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1
        for label, key, dv in (("Length", "lx", 0.30), ("Width", "ly", 0.10), ("Height", "lz", 0.08)):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=str(dv)); self.vars[key] = v
            ttk.Entry(form, textvariable=v, width=8).grid(row=r, column=1, sticky="w")
            ttk.Label(form, text="m").grid(row=r, column=2, sticky="w"); r += 1
        ttk.Label(form, text="CAD file units").grid(row=r, column=0, sticky="w")
        self.units_var = tk.StringVar(value="mm")
        ttk.Combobox(form, textvariable=self.units_var, values=["mm", "cm", "m", "in"],
                     width=5, state="readonly").grid(row=r, column=1, sticky="w"); r += 1
        ttk.Button(form, text="Import CAD volume…", command=self.on_import).grid(
            row=r, column=0, columnspan=3, sticky="ew", pady=(2, 0)); r += 1

        ttk.Label(form, text="Condition", font=("", 10, "bold")).grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(8, 0)); r += 1
        for label, key, dv, unit in (("Airspeed", "V", 20.0, "m/s"), ("Target lift", "target", 20.0, "N")):
            ttk.Label(form, text=label).grid(row=r, column=0, sticky="w")
            v = tk.StringVar(value=str(dv)); self.vars[key] = v
            ttk.Entry(form, textvariable=v, width=8).grid(row=r, column=1, sticky="w")
            ttk.Label(form, text=unit).grid(row=r, column=2, sticky="w"); r += 1

        self.run_btn = ttk.Button(form, text="Design it", command=self.on_run)
        self.run_btn.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(12, 4)); r += 1
        self.save_btn = ttk.Button(form, text="Save CAD…", command=self.on_save, state="disabled")
        self.save_btn.grid(row=r, column=0, columnspan=3, sticky="ew"); r += 1
        self.status = ttk.Label(form, text="Set inputs and press Design it.", foreground="#555")
        self.status.grid(row=r, column=0, columnspan=3, sticky="w", pady=(10, 0))

        self.fig = Figure(figsize=(5.6, 4.2), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d"); self._blank()
        self.canvas = FigureCanvasTkAgg(self.fig, master=main)
        self.canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(1, weight=1); main.rowconfigure(0, weight=1)
        self.out = tk.Text(main, height=7, width=90, wrap="word", state="disabled",
                           font=("Menlo", 10))
        self.out.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.root.after(100, self._poll)

    def _on_type(self, *a):
        t = LABEL_TYPE[self.type_var.get()]
        objs = [OBJ_LABEL[o] for o in TYPES[t]["objectives"]]
        self.obj_cb["values"] = objs
        if self.obj_var.get() not in objs:
            self.obj_var.set(objs[0])

    def _blank(self):
        self.ax.clear()
        self.ax.set_title("Designed aircraft (drag to rotate)")
        self.ax.set_xlabel("x"); self.ax.set_ylabel("y"); self.ax.set_zlabel("z")

    def on_import(self):
        path = filedialog.askopenfilename(
            title="Import a CAD file (the volume your parts occupy)",
            filetypes=[("CAD / mesh",
                        "*.step *.stp *.iges *.igs *.brep *.stl *.obj *.ply *.off *.glb *.3mf"),
                       ("All files", "*.*")])
        if not path:
            return
        try:
            b = cad3d.import_volume(path, units=self.units_var.get())
            self.vars["lx"].set(f"{b.lx:.4g}"); self.vars["ly"].set(f"{b.ly:.4g}")
            self.vars["lz"].set(f"{b.lz:.4g}")
            self.status.config(text=f"Imported {b.lx:.3f}×{b.ly:.3f}×{b.lz:.3f} m "
                                    f"(from {self.units_var.get()}).")
        except Exception as e:  # noqa
            messagebox.showerror("Import failed", str(e))

    def on_run(self):
        try:
            spec = DesignSpec(
                box=Box(float(self.vars["lx"].get()), float(self.vars["ly"].get()),
                        float(self.vars["lz"].get())),
                V=float(self.vars["V"].get()),
                aircraft_type=LABEL_TYPE[self.type_var.get()],
                objective=LABEL_OBJ[self.obj_var.get()],
                target_lift_N=float(self.vars["target"].get()))
            assert spec.box.lx > 0 and spec.box.ly > 0 and spec.box.lz > 0 and spec.V > 0
        except Exception:
            messagebox.showerror("Invalid input", "Enter positive numbers for all fields."); return
        self.run_btn.config(state="disabled"); self.save_btn.config(state="disabled")
        self.status.config(text="Designing… (hundreds of aero evaluations, ~30–90 s)")
        threading.Thread(target=self._work, args=(spec,), daemon=True).start()

    def _work(self, spec):
        try:
            self.q.put(("done", design(spec, maxiter=12)))
        except Exception as e:  # noqa
            self.q.put(("err", str(e)))

    def _poll(self):
        try:
            k, p = self.q.get_nowait()
            if k == "done":
                self._show(p)
            else:
                self.run_btn.config(state="normal"); self.status.config(text="Error — see dialog.")
                messagebox.showerror("Design failed", p)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _show(self, d):
        self.design = d
        self.run_btn.config(state="normal")
        self._blank()
        try:
            m = aircraft_mesh(d); v = m.vertices
            self.ax.plot_trisurf(v[:, 0], v[:, 1], v[:, 2], triangles=m.faces,
                                 color="#16a34a", alpha=0.55, linewidth=0)
            self.ax.set_box_aspect((np.ptp(v[:, 0]) or 1, np.ptp(v[:, 1]) or 1, np.ptp(v[:, 2]) or 1))
        except Exception:
            pass
        self.canvas.draw()
        feats = "  ".join(f"{k}={val:.3f}" for k, val in d.params.items())
        lift = f"{d.lift_N:.1f} N" if np.isfinite(d.lift_N) else "n/a (rotor-lifted)"
        ld = f"{d.LD:.1f}" if np.isfinite(d.LD) else "n/a"
        lines = [
            f"{TYPE_LABEL[d.aircraft_type]}  —  {OBJ_LABEL[d.objective]}",
            f"designed features: {feats}",
            f"lift {lift}    drag {d.drag_N:.2f} N    L/D {ld}",
            "",
            "Aero: AeroSandbox design methods. Your CFD-trained 3D surrogate (Stage B) swaps in later.",
        ]
        self.out.config(state="normal"); self.out.delete("1.0", "end")
        self.out.insert("1.0", "\n".join(lines)); self.out.config(state="disabled")
        self.status.config(text="Done. Save the aircraft as CAD if you like.")
        self.save_btn.config(state="normal")

    def on_save(self):
        if self.design is None:
            return
        path = filedialog.asksaveasfilename(defaultextension=".stl",
                                            filetypes=[("STL", "*.stl"), ("OBJ", "*.obj"),
                                                       ("PLY", "*.ply")])
        if not path:
            return
        try:
            export_aircraft(self.design, path)
            messagebox.showinfo("Saved", f"Wrote {path}")
        except Exception as e:  # noqa
            messagebox.showerror("Export failed", str(e))


def main():
    root = tk.Tk()
    AircraftGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
