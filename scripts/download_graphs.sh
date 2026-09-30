#!/bin/bash
# Resumable download of the Z-Image-Turbo LiteRT graphs.
mkdir -p "$(dirname "$0")/../models/litert" && cd "$(dirname "$0")/../models/litert"
BASE=https://huggingface.co/litert-community/Z-Image-Turbo-LiteRT/resolve/09ea3ae2ef44d04d0ad1591de36d1c13504e1521
for f in z_refx z_refc zc_final zvae z_embx z_embc zc_main0 zc_main1 zc_main2 zc_main3 zc_main4 zc_main5 qwen_enc; do
  echo "$(date +%T) $f"
  curl -sSL --retry 5 -C - -o "$f.tflite" "$BASE/$f.tflite" || echo "FAILED $f"
done
echo "$(date +%T) DONE"
