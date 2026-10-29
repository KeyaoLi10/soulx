#!/bin/bash
# Fetch wheels from the tsinghua PyPI mirror by parsing /simple/ index pages.
# Usage: fetch_wheels.sh PKG==VER [PKG==VER ...]   -> /tmp/wheels/
set -u
OUT=/tmp/wheels
mkdir -p $OUT
IDX=https://pypi.tuna.tsinghua.edu.cn/simple

for spec in "$@"; do
  pkg="${spec%%=*}"; ver="${spec##*=}"
  [ "$pkg" = "$ver" ] && { echo "need PKG==VER for $spec"; exit 1; }
  name=$(echo "$pkg" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
  # match the wheel: cp310, linux x86_64, py3-none-any as fallback
  urls=$(timeout 30 curl -sL "$IDX/$name/" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//')
  pick=""
  for pat in "cp310-cp310-manylinux" "cp310-cp310-linux_x86_64" "py3-none-any" "cp310-any"; do
    pick=$(echo "$urls" | grep -E "/${pkg//-/_}-${ver}-[^/]*${pat}.*\.(whl|tar\.gz)#" | head -1)
    [ -n "$pick" ] && break
  done
  if [ -z "$pick" ]; then
    # some pkgs publish e.g. manylinux_2_17_x86_64.manylinux2014 style; retry loose
    pick=$(echo "$urls" | grep -E "/${pkg//-/_}-${ver}-" | grep -E "cp310|py3-none" | grep "\.whl#" | head -1)
  fi
  [ -z "$pick" ] && { echo "NOT FOUND: $spec"; continue; }
  url="https://pypi.tuna.tsinghua.edu.cn/${pick#../../}"
  fname=$(basename "${url%%#*}")
  [ -f "$OUT/$fname" ] && { echo "HAVE $fname"; continue; }
  for a in 1 2 3; do
    curl -sSL -C - --retry 3 -o "$OUT/$fname" "$url" && break
    sleep 2
  done
  echo "GOT $fname ($(stat -c %s "$OUT/$fname") bytes)"
done
