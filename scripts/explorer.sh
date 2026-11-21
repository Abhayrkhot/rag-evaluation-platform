#!/usr/bin/env bash
# Mac-side helper for Explorer (git lives on the box; Explorer gets `git archive` copies).
#   scripts/explorer.sh push [ref]     copy a commit to /scratch/$USER/rag/src/<sha> (+ GIT_SHA), link src/current
#   scripts/explorer.sh data           copy data/hotpot100k from the box to /scratch/$USER/rag/data
#   scripts/explorer.sh run '<cmd>'    run a shell command on the login node (in src/current)
#   scripts/explorer.sh pull           copy /scratch/$USER/rag/results/ to the box repo's results/explorer/
# Connection details come from the environment or an untracked .explorer.env next to the repo root:
#   EXPLORER_HOST=<user>@login.explorer.northeastern.edu   EXPLORER_KEY=<path to the RC-issued key>
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$here/.explorer.env" ] && source "$here/.explorer.env"
: "${EXPLORER_HOST:?set EXPLORER_HOST (see header)}"
EXP=(ssh ${EXPLORER_KEY:+-i "$EXPLORER_KEY"} -o ServerAliveInterval=15 -o ConnectTimeout=90 "$EXPLORER_HOST")
REPO='~/Projects/rag-evaluation-platform'
cmd=${1:-}; shift || true
case "$cmd" in
  push)
    ref=${1:-HEAD}
    sha=$(ssh box "cd $REPO && git rev-parse --short=12 $ref")
    ssh box "cd $REPO && git archive --format=tar $ref" | "${EXP[@]}" "set -e
      d=/scratch/\$USER/rag/src/$sha; mkdir -p \$d /scratch/\$USER/rag/{data,artifacts,results,logs}
      tar xf - -C \$d; echo $sha > \$d/GIT_SHA
      for x in data artifacts results; do rm -rf \$d/\$x; ln -sfn /scratch/\$USER/rag/\$x \$d/\$x; done
      ln -sfn \$d /scratch/\$USER/rag/src/current; echo pushed \$d"
    ;;
  data)
    ssh box "tar czf - -C $REPO data/hotpot100k" | "${EXP[@]}" 'set -e; mkdir -p /scratch/$USER/rag; tar xzf - -C /scratch/$USER/rag; md5sum /scratch/$USER/rag/data/hotpot100k/*.jsonl'
    ssh box "cd $REPO && md5sum data/hotpot100k/*.jsonl"
    ;;
  run)
    "${EXP[@]}" "cd \$(readlink -f /scratch/\$USER/rag/src/current) && $*"
    ;;
  pull)
    "${EXP[@]}" 'tar czf - -C /scratch/$USER/rag/results .' | ssh box "mkdir -p $REPO/results/explorer && tar xzf - -C $REPO/results/explorer"
    ;;
  *) echo "usage: $0 push [ref] | data | run '<cmd>' | pull" >&2; exit 2 ;;
esac
