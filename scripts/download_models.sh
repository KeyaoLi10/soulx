#!/bin/bash
# Parallel (range-split) download of SoulX-Duplug-0.6B from HF via mihomo :17891.
# Single connection ~1MB/s through this proxy; N parallel ranges give ~N MB/s.
set -u
PROXY=""
BASE="https://hf-mirror.com/Soul-AILab/SoulX-Duplug-0.6B/resolve/main"
DEST=/home/research/krisl/soulx/pretrained_models
TOKEN=$(cat /home/research/krisl/hf_token.txt | tr -d '[:space:]')
NPARTS=12

fetch_small() { # fetch_small <path>
  local f="$1" out="$DEST/$1"
  mkdir -p "$(dirname "$out")"
  [ -f "$out.ok" ] && { echo "SKIP $f"; return; }
  for a in 1 2 3; do
    curl -sS ${PROXY:+-x "$PROXY"} -L --retry 3 --retry-delay 2 \
      -H "Authorization: Bearer $TOKEN" -o "$out" "$BASE/$f" && break
    sleep 3
  done
  touch "$out.ok"; echo "DONE $f"
}

fetch_big() { # fetch_big <path> <size>
  local f="$1" size="$2" out="$DEST/$1"
  mkdir -p "$(dirname "$out")"
  [ -f "$out.ok" ] && { echo "SKIP $f"; return; }
  local chunk=$(( (size + NPARTS - 1) / NPARTS ))
  local pids=()
  for i in $(seq 0 $((NPARTS-1))); do
    (
      local_start=$((i*chunk)); local_end=$(( (i+1)*chunk - 1 ))
      [ $local_end -ge $size ] && local_end=$((size-1))
      [ $local_start -gt $local_end ] && exit 0
      part="$out.part$i"
      for a in 1 2 3 4 5 6 7 8; do
        curl -sS ${PROXY:+-x "$PROXY"} -L -C - --retry 3 --retry-delay 2 \
          -H "Authorization: Bearer $TOKEN" \
          -r "$local_start-$local_end" -o "$part" "$BASE/$f" && break
        sleep 3
      done
    ) &
    pids+=($!)
  done
  wait "${pids[@]}"
  # verify + assemble
  local total=0 parts=()
  for i in $(seq 0 $((NPARTS-1))); do
    part="$out.part$i"; [ -f "$part" ] || continue
    parts+=("$part")
    total=$(( total + $(stat -c %s "$part") ))
  done
  if [ "$total" -ne "$size" ]; then echo "SIZE MISMATCH $f: $total != $size"; exit 1; fi
  cat "${parts[@]}" > "$out"
  rm -f "$out".part*
  touch "$out.ok"; echo "DONE $f ($size bytes)"
}

fetch_big "Qwen3-0.6B-expand_vocab_v2/model.safetensors" 2384234968
fetch_big "SoulX-Duplug/SoulX-Duplug-0.6B-Bilingual.pth" 3896904668
fetch_big "glm-4-voice-tokenizer/model.safetensors" 1458374480

for f in \
  "Qwen3-0.6B-expand_vocab_v2/added_tokens.json" \
  "Qwen3-0.6B-expand_vocab_v2/chat_template.jinja" \
  "Qwen3-0.6B-expand_vocab_v2/config.json" \
  "Qwen3-0.6B-expand_vocab_v2/config_record.yaml" \
  "Qwen3-0.6B-expand_vocab_v2/generation_config.json" \
  "Qwen3-0.6B-expand_vocab_v2/merges.txt" \
  "Qwen3-0.6B-expand_vocab_v2/special_tokens_map.json" \
  "Qwen3-0.6B-expand_vocab_v2/tokenizer.json" \
  "Qwen3-0.6B-expand_vocab_v2/tokenizer_config.json" \
  "Qwen3-0.6B-expand_vocab_v2/vocab.json" \
  "config.yaml" \
  "glm-4-voice-tokenizer/.msc" \
  "glm-4-voice-tokenizer/.mv" \
  "glm-4-voice-tokenizer/config.json" \
  "glm-4-voice-tokenizer/configuration.json" \
  "glm-4-voice-tokenizer/preprocessor_config.json" \
; do
  fetch_small "$f"
done
echo "ALL DONE"
