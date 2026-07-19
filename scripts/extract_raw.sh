#!/bin/sh
# Extract the five dataset archives into data/raw and lock it read-only.
# Idempotent: an already-extracted dataset is skipped (raw is sacred — to
# re-extract, delete the target directory by hand first).
set -eu
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ZIP_DIR="${BVIZ_ZIP_DIR:-$HOME/Downloads}"
RAW="$ROOT/data/raw"

extract() {
    zip_name="$1"
    dest="$2"
    if [ -d "$dest" ]; then
        echo "skip: $dest already exists"
        return
    fi
    mkdir -p "$dest"
    unzip -q "$ZIP_DIR/$zip_name" -d "$dest"
    echo "extracted: $zip_name -> $dest"
}

chmod -R u+w "$RAW" 2>/dev/null || true
extract ShuttleSet.zip "$RAW/shuttleset"
extract ShuttleSet22_CoachAI_Challenge.zip "$RAW/shuttleset22"
extract BadmintonDB.zip "$RAW/badminton_db"
extract TrackNetV3_code.zip "$RAW/tracknetv3"
extract RacketDB_annotations.zip "$RAW/racketdb"
chmod -R a-w "$RAW"
echo "data/raw locked read-only"
