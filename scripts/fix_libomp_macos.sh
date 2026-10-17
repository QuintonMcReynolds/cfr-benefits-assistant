#!/bin/sh
# macOS without Homebrew: xgboost/lightgbm wheels look for libomp in /opt/homebrew. Reuse the copy torch ships.
set -e
SP=$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])")
for f in "$SP/xgboost/lib/libxgboost.dylib" "$SP/lightgbm/lib/lib_lightgbm.dylib"; do
  [ -f "$f" ] || continue
  ln -sf "$SP/torch/lib/libomp.dylib" "$(dirname "$f")/libomp.dylib"
  install_name_tool -add_rpath @loader_path "$f" 2>/dev/null || true
  codesign -f -s - "$f"
done
