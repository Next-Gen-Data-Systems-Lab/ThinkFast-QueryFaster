#!/usr/bin/env bash
# Copy what the job needs to the CHTC access point. Does not submit anything.
#
#   bash chtc/upload.sh
#
# CHTC_HOST defaults to the "chtc" SSH alias. REMOTE_DIR is relative to your
# home directory on the access point unless it starts with /.
set -euo pipefail

HOST=${CHTC_HOST:-chtc}
REMOTE_DIR=${REMOTE_DIR:-ThinkFast-QueryFaster}
PROJECT=$(cd "$(dirname "$0")/.." && pwd)

cd "$PROJECT"
ssh "$HOST" "mkdir -p '$REMOTE_DIR/logs'"
# Named files only, so chtc/.env never leaves this machine.
rsync -aR --exclude='__pycache__' --exclude='*.pyc' \
    src chtc/run.sh chtc/run_kev.sh chtc/run_jev.sh chtc/run_scan.sh \
    chtc/midterm.sub chtc/kev.sub chtc/jev.sub chtc/scan.sub requirements.txt \
    "$HOST:$REMOTE_DIR/"
# Built data (src.synergy_build). scan.sub transfers data/, so it must exist.
ssh "$HOST" "mkdir -p '$REMOTE_DIR/data'"
# data/arxiv is left out: jobs build the arXiv sample from the pinned Hugging
# Face snapshot, which is faster than shipping a gigabyte with every job.
if [ -d data ]; then
    rsync -a --exclude='arxiv/' data/ "$HOST:$REMOTE_DIR/data/"
fi

echo "Uploaded to $HOST:$REMOTE_DIR. To submit:"
echo "  ssh $HOST"
echo "  cd $REMOTE_DIR && condor_submit chtc/midterm.sub"
