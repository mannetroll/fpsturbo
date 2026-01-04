#!/usr/bin/env bash
set -euo pipefail

#
#  From fps/:
#    $ nohup stdbuf -oL bash fps_sweep_cufft.sh > fps_sweep_cufft.txt 2>&1 &
#
#    $ PYTHONPATH=.. uv run python -m scipyturbo.turbo_simulator_cufft 512
#

rm -f fps_N*.log
OUT_CSV="fps_sweep_cufft.csv"
rm -f "${OUT_CSV}"

# Header
echo "N,FPS" > "${OUT_CSV}"

for N in 32 64 128 256 512 1024 2048 3072 4096 8192 9216; do
    LOG="fps_N${N}.log"
    echo "Running N=${N} ..."

    # Run and capture full output (run from fps/, so PYTHONPATH=..)
    stdbuf -oL env PYTHONPATH=.. uv run python -m scipyturbo.turbo_simulator_cufft "${N}" | tee "${LOG}"

    # Extract FPS from:  " FPS = 127.447"
    FPS=$(grep -E "FPS =" "${LOG}" | tail -n 1 | awk '{print $3}')

    echo "${N},${FPS}" >> "${OUT_CSV}"
done

echo "Done. Results written to ${OUT_CSV}"