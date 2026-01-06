#!/usr/bin/env bash
set -euo pipefail

#
#  From fps/:
#    $ nohup stdbuf -oL bash fps_sweep_cufft.sh > fps_sweep_cufft.txt 2>&1 &
#
#    $ PYTHONPATH=.. uv run python -m scipyturbo.turbo_simulator_cufft 512 1E4 10 1001 0.5 gpu
#

rm -f fps_N*.log
OUT_CSV="fps_sweep_cufft.csv"
rm -f "${OUT_CSV}"

# Header
echo "N,FPS" > "${OUT_CSV}"

for N in 32 64 128 256 384 400 405 432 450 480 486 500 512 540 576 600 625 640 648 675 720 729 750 768 800 810 864 900 960 972 1000 1024 1536 2048 3072 4096 8192 9216; do
    LOG="fps_N${N}.log"
    echo "Running N=${N} ..."

    # Run and capture full output (run from fps/, so PYTHONPATH=..)
    PYTHONPATH=.. PYTHONUNBUFFERED=1 uv run python -u -m scipyturbo.turbo_simulator_cufft "$N" 1e4 10 5001 2>&1 | stdbuf -oL -eL tee -a "$LOG"

    # Extract FPS from:  " FPS = 127.447"
    FPS=$(grep -E "FPS =" "${LOG}" | tail -n 1 | awk '{print $3}')

    echo "${N},${FPS}" >> "${OUT_CSV}"
done

echo "Done. Results written to ${OUT_CSV}"