#!/usr/bin/env bash
# Re-optimize benchmark figures for the web AFTER (re)generating them.
#
# matplotlib emits large 24-bit PNGs (some >2000 px wide / ~200 KB). This caps
# the width to 1400 px (plenty for the furo content column at 2x), strips
# metadata, and quantizes to a 256-colour palette -- visually lossless for the
# line/scatter/spy plots here, typically ~55% smaller overall.
#
# Run after any benchmark regeneration:
#     bash benchmarks/optimize_figures.sh
#
# (For true vector output, have the plotting scripts also savefig(..., format=
# "svg") -- the per-op results are cached as JSON in assets/benchmarks/, so the
# figures can be re-plotted to SVG without re-running the benchmarks.)
set -euo pipefail
dir="${1:-$(cd "$(dirname "$0")/../assets/benchmarks" && pwd)}"
command -v convert >/dev/null || { echo "needs ImageMagick (convert)"; exit 1; }

before=$(du -sk "$dir" | cut -f1); n=0
while IFS= read -r -d '' f; do
  convert "$f" -strip -resize '1400x1400>' -colors 256 \
          -define png:compression-level=9 "PNG8:$f.tmp" 2>/dev/null || { rm -f "$f.tmp"; continue; }
  if [ "$(stat -c%s "$f.tmp")" -lt "$(stat -c%s "$f")" ]; then mv "$f.tmp" "$f"; n=$((n+1)); else rm -f "$f.tmp"; fi
done < <(find "$dir" -name '*.png' -print0)
after=$(du -sk "$dir" | cut -f1)
echo "optimized $n PNGs in $dir: ${before}K -> ${after}K"
