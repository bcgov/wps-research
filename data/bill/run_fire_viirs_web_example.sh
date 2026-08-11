#!/bin/bash
# Launch the user-defined VIIRS fire-mapping web app.
# Sibling of run_fire_web.sh — no polygon shapefile; analysts create
# fires by drawing a bbox + date range in /new_fire.

RASTERS=(
    # /ram/new_cloudfree/pgfc_2023.bin
    # /ram/new_cloudfree/2024_pgfc.bin
    # /ram/new_cloudfree/2025_pgfc.bin
    # /ram/konni/2026_konni.bin
    /data/mrap_bc/20260810_mrap.bin
)

OUT_ROOT="./fire_mapping_results_viirs"

# LAADS DAAC token (one line, your token). Default location.
LAADS_TOKEN_FILE="/data/.tokens/laads"

python3 -m batch_fire_mapping_viirs_web \
    --rasters "${RASTERS[@]}" \
    --out_root "$OUT_ROOT" \
    --laads_token_file "$LAADS_TOKEN_FILE" \
    --user_password password_goes_here \
    --admin_password admin_password_goes_here
    #--disable_overview_force_regeneration
    # //--skip_viirs_bootstrap \
    #--disable_overview_force_regeneration
