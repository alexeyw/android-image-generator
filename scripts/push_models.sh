#!/bin/bash
# Push the Mac-side model files to the phone instead of downloading them on the device.
#   scripts/download_graphs.sh && .venv/bin/python reference/host_assets.py   # once, on the Mac
#   scripts/push_models.sh [adb-serial]
set -euo pipefail
cd "$(dirname "$0")/.."
ADB_BIN=$(command -v adb || echo "${ANDROID_HOME:-$HOME/Library/Android/sdk}/platform-tools/adb")
ADB=("$ADB_BIN" ${1:+-s "$1"})
PKG=io.github.alexeyw.zimage
DST=/sdcard/Android/data/$PKG/files/models
# The app must create its own directories: ones made by `adb shell mkdir` under Android/data
# belong to the shell user (mode 2770) and the app cannot read them.
"${ADB[@]}" shell am start -n $PKG/.MainActivity >/dev/null
for _ in $(seq 1 30); do
  "${ADB[@]}" shell "[ -d $DST/litert ] && [ -d $DST/host ]" && break
  sleep 1
done
"${ADB[@]}" shell "[ -d $DST/litert ] && [ -d $DST/host ]" || { echo "app did not create $DST (installed?)"; exit 1; }
for f in models/litert/*.tflite; do
  "${ADB[@]}" push "$f" "$DST/litert/"
done
for f in vocab.json merges.txt t_emb_w1.f32 t_emb_b1.f32 t_emb_w2.f32 t_emb_b2.f32 cap_pad_token.f32 embed_tokens.bf16; do
  "${ADB[@]}" push "models/host/$f" "$DST/host/"
done
echo "done: $DST"
