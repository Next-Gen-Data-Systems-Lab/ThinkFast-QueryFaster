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
    src chtc/run.sh chtc/midterm.sub requirements.txt \
    "$HOST:$REMOTE_DIR/"

echo "Uploaded to $HOST:$REMOTE_DIR. To submit:"
echo "  ssh $HOST"
echo "  cd $REMOTE_DIR && condor_submit chtc/midterm.sub"
