#!/bin/bash
#SBATCH --job-name=mjx_ftzband
#SBATCH --partition=compute
#SBATCH --account=ab0995
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%x-%j.out
#
# The FTZ oracle vs the standard oracle of one variant (plan decision 13; reference/ftz_band.py):
#   sbatch reference/jobs/ftz_band.sbatch MJX_COMMIT EXP INPUT_DIR STD_JOB FTZ_JOB
# compares the invisibility triples job<STD_JOB>-{plain,jdon} (standard binaries) and job<FTZ_JOB>-{plain,jdon}
# (their -ftz variants, reference/relink_ftz.sh) under $MJX_REFERENCE_RUNS/EXP/INPUT_DIR with the frozen scripts of
# MJX_COMMIT; writes ftzband-job<FTZ_JOB>-{plain,jdon}.{txt,json} next to the runs (refuses existing ones).
# Verdict lines "BAND CONFINED ..." / "OUTSIDE BAND ..."; exit 1 if any comparison is outside the band.
# Never deletes anything.
set -u
SELF=$0; [ -f "$(dirname "$SELF")/../../mitjax/paths.py" ] ||
  SELF=$(scontrol show job "${SLURM_JOB_ID:-none}" 2>/dev/null | sed -n 's/^ *Command=//p' | cut -d' ' -f1)
REPO=${MJX_TREE:-$(cd "$(dirname "$SELF")/../.." && pwd)}
[ -f "$REPO/mitjax/paths.py" ] || { echo "FAIL: no mitjax repository at $REPO (set MJX_TREE)"; exit 1; }
[ -f "$REPO/levante.env" ] && . "$REPO/levante.env"   # machine paths (mitjax/paths.py is neutral)
PY=${MJX_PYTHON:-python}
eval "$("$PY" "$REPO/mitjax/paths.py" --sh)"
COMMIT=$(git -C "$REPO" rev-parse --verify "${1:?mitjax commit}^{commit}") || { echo "FAIL: no commit $1"; exit 1; }
EXP=${2:?experiment}; INP=${3:?input dir}; SJ=${4:?standard job}; FJ=${5:?ftz job}
echo "job ${SLURM_JOB_ID:-none} $(hostname) $(date -Is) repo $REPO mitjax $COMMIT"
SNAP=$MJX_RUNS/snapshots/${SLURM_JOB_NAME:-mjx_ftzband}-${SLURM_JOB_ID:?run this through sbatch}
[ -e "$SNAP" ] && { echo "FAIL: $SNAP exists"; exit 1; }
mkdir -p "$SNAP"
git -C "$REPO" archive --format=tar "$COMMIT" reference mitjax/paths.py $(git -C "$REPO" ls-tree --name-only "$COMMIT" mitjax/make_rundir.py mitjax/testreport_jax.py) mitjax/io/dump.py | tar -x -C "$SNAP"
echo "$COMMIT" > "$SNAP/COMMIT"
BASE=$MJX_REFERENCE_RUNS/$EXP/$INP
FAILED=0
for MODE in plain jdon; do
  OUT=$BASE/ftzband-job$FJ-$MODE
  [ -e "$OUT.txt" ] || [ -e "$OUT.json" ] && { echo "FAIL: $OUT.* exists"; FAILED=1; continue; }
  "$PY" "$SNAP/reference/ftz_band.py" "$BASE/job$SJ-$MODE" "$BASE/job$FJ-$MODE" --out "$OUT.txt" --json "$OUT.json" \
    | grep -E "^(BAND CONFINED|OUTSIDE BAND|  (STDOUT|dumps|written|file|dump|text|problem))" || true
  grep -q "^BAND CONFINED" "$OUT.txt" 2>/dev/null || FAILED=1
done
[ "$FAILED" -eq 0 ] || { echo "FTZ BAND: OUTSIDE OR FAILED"; exit 1; }
echo "FTZ BAND: ALL CONFINED"
