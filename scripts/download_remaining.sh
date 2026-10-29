#!/bin/bash
# Download the remaining glm-4-voice-tokenizer files (previous run hit an
# hf-mirror outage). hf-mirror primary, mihomo :17891 fallback per part.
set -u
TOKEN=$(cat /home/research/krisl/hf_token.txt | tr -d '[:space:]')
DEST=/home/research/krisl/soulx/pretrained_models
NPARTS=6
SIZE=1458374480
REL="glm-4-voice-tokenizer/model.safetensors"

MIRROR="https://hf-mirror.com/Soul-AILab/SoulX-Duplug-0.6B/resolve/main"
PROXY_URL="https://huggingface.co/Soul-AILab/SoulX-Duplug-0.6B/resolve/main"

out="$DEST/$REL"
mkdir -p "$(dirname "$out")"
chunk=$(( (SIZE + NPARTS - 1) / NPARTS ))
pids=()
for i in $(seq 0 $((NPARTS-1))); do
  (
    s=$((i*chunk)); e=$(( (i+1)*chunk - 1 )); [ $e -ge $SIZE ] && e=$((SIZE-1))
    part="$out.part$i"
    for a in $(seq 1 20); do
      # try mirror first, then the slow proxy
      for base in "$MIRROR" "$PROXY_URL"; do
        [ "$base" = "$PROXY_URL" ] && px=(-x http://127.0.0.1:17891) || px=()
        curl -sS "${px[@]}" -L -C - --retry 2 --retry-delay 2 \
          -H "Authorization: Bearer $TOKEN" \
          -r "$s-$e" -o "$part" "$base/$REL" 2>/dev/null
        sz=0; [ -f "$part" ] && sz=$(stat -c %s "$part")
        [ "$sz" -ge $((e - s + 1)) ] && exit 0
      done
      sleep 5
    done
    exit 1
  ) &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 1 ] && { echo "PARTS FAILED"; exit 1; }
total=0; parts=()
for i in $(seq 0 $((NPARTS-1))); do
  part="$out.part$i"; parts+=("$part"); total=$(( total + $(stat -c %s "$part") ))
done
[ "$total" -ne "$SIZE" ] && { echo "SIZE MISMATCH $total != $SIZE"; exit 1; }
cat "${parts[@]}" > "$out" && rm -f "$out".part*
echo "DONE $REL"

# small files for glm-4-voice-tokenizer + top-level config.yaml
for f in ".msc" ".mv" "config.json" "configuration.json" "preprocessor_config.json"; do
  out2="$DEST/glm-4-voice-tokenizer/$f"
  for a in 1 2 3 4 5; do
    curl -sS -L --retry 2 -H "Authorization: Bearer $TOKEN" \
      -o "$out2" "$MIRROR/glm-4-voice-tokenizer/$f" 2>/dev/null && \
      [ -s "$out2" ] && break
    curl -sS -x http://127.0.0.1:17891 -L --retry 2 \
      -H "Authorization: Bearer $TOKEN" \
      -o "$out2" "$PROXY_URL/glm-4-voice-tokenizer/$f" 2>/dev/null && \
      [ -s "$out2" ] && break
    sleep 3
  done
  echo "DONE glm-4-voice-tokenizer/$f ($(stat -c %s "$out2" 2>/dev/null || echo 0) bytes)"
done
curl -sS -L -o "$DEST/config.yaml" "$MIRROR/config.yaml" 2>/dev/null || \
  curl -sS -x http://127.0.0.1:17891 -L -o "$DEST/config.yaml" "$PROXY_URL/config.yaml"
echo "ALL REMAINING DONE"
