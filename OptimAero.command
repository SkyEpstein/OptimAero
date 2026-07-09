#!/bin/bash
# OptimAero — Aerodynamic Shape Optimizer.  Double-click this file to launch the app.
cd /Users/skyepstein/OptimAero || { echo "OptimAero folder not found."; read -r; exit 1; }
echo "Launching OptimAero — Aerodynamic Shape Optimizer…"
echo "(this window can be minimized; close it to quit the app)"
.venv/bin/python -m optimaero.gui_shapeopt
code=$?
if [ $code -ne 0 ]; then
  echo ""
  echo "OptimAero exited with an error (code $code). Press Enter to close."
  read -r
fi
