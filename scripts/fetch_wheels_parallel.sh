#!/bin/bash
# Parallel-range wheel fetcher from the tsinghua PyPI mirror.
# The mirror serves ~17MB/s per ranged connection but throttles sustained
# single streams, so split each wheel into NPARTS ranges.
# Usage: fetch_wheels_parallel.sh PKG==VER [PKG==VER ...]  -> /tmp/wheels/
set -u
OUT=/tmp/wheels
IDX=https://pypi.tuna.tsinghua.edu.cn/simple
NPARTS=8
mkdir -p $OUT

resolve() { # resolve PKG VER -> echo url
  local pkg="$1" ver="$2"
  local name=$(echo "$pkg" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  local urls=$(timeout 30 curl -sL "$IDX/$name/" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//')
  local pick=""
  for pat in "cp310-cp310-manylinux1_x86_64" "cp310-cp310-manylinux.*x86_64" \
             "py3-none-manylinux.*x86_64" "cp310-cp310-linux_x86_64" \
             "py3-none-any" "cp310-any" "cp310-cp310-musllinux.*x86_64"; do
    pick=$(echo "$urls" | grep -E "/${pkg//-/_}-${ver}-[^/]*${pat}[^/]*\.whl#" | head -1)
    [ -n "$pick" ] && break
  done
  [ -z "$pick" ] && { echo ""; return; }
  echo "https://pypi.tuna.tsinghua.edu.cn/${pick#../../}"
}

fetch_one() { # fetch_one URL FNAME
  local url="$1" fname="$2" out="$OUT/$fname"
  [ -f "$out.done" ] && { echo "HAVE $fname"; return; }
  local size=$(timeout 30 curl -sIL "$url" 2>/dev/null | grep -i "^content-length" | tail -1 | tr -dc '0-9')
  if [ -z "$size" ] || [ "$size" -lt 20000000 ]; then
    # small file: single shot
    curl -sSL --retry 3 -o "$out" "$url" && touch "$out.done" && { echo "GOT $fname ($size)"; return; }
  fi
  local chunk=$(( (size + NPARTS - 1) / NPARTS ))
  local pids=()
  for i in $(seq 0 $((NPARTS-1))); do
    (
      s=$((i*chunk)); e=$(( (i+1)*chunk - 1 )); [ $e -ge $size ] && e=$((size-1))
      [ $s -gt $e ] && exit 0
      part="$out.part$i"
      for a in $(seq 1 12); do
        curl -sS -L -C - --retry 2 -r "$s-$e" -o "$part" "$url" 2>/dev/null
        sz=0; [ -f "$part" ] && sz=$(stat -c %s "$part")
        [ "$sz" -ge $((e - s + 1)) ] && exit 0
        sleep 2
      done
      exit 1
    ) &
    pids+=($!)
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ "$fail" -eq 1 ] && { echo "FAILED $fname"; return; }
  local parts=()
  for i in $(seq 0 $((NPARTS-1))); do [ -f "$out.part$i" ] && parts+=("$out.part$i"); done
  cat "${parts[@]}" > "$out" && rm -f "$out".part*
  touch "$out.done"
  echo "GOT $fname ($(stat -c %s "$out") bytes)"
}

for spec in "$@"; do
  pkg="${spec%%=*}"; ver="${spec##*=}"
  url=$(resolve "$pkg" "$ver")
  if [ -z "$url" ]; then echo "NOT FOUND: $spec"; continue; fi
  fname=$(basename "${url%%#*}")
  fetch_one "$url" "$fname"
done
echo "WHEELS ALL DONE"
