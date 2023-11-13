#!/usr/bin/bash

### DOWNLOAD VOICE SAMPLES ###
BLOB_NAMES=(
    clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2
)
AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

DATA_DIR="data"
OUTPUT_PATH="$DATA_DIR/datasets_fullband"

mkdir -p $OUTPUT_PATH/{clean_fullband,noise_fullband}

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"
    curl "$URL" | tar -C "$OUTPUT_PATH" -f - -x -j
done

## DOWNLOAD NOISE SAMPLES ###
repo_subdir="noise_train"

git -C "$DATA_DIR" init
git -C "$DATA_DIR" config core.sparseCheckout true
echo "noise_train/*" > "$DATA_DIR/.git/info/sparse-checkout"
git -C "$DATA_DIR" remote add -f origin https://github.com/microsoft/MS-SNSD.git
git -C "$DATA_DIR" pull origin master
rm -rf "$DATA_DIR/.git"

### DOWNLOAD RIRS ###
openslr_url="https://www.openslr.org/resources/28/rirs_noises.zip"
openslr_dir="$DATA_DIR/rirs_noises"
mkdir -p "$openslr_dir"
wget -O "$openslr_dir/rirs_noises.zip" "$openslr_url"
unzip "$openslr_dir/rirs_noises.zip" -d "$openslr_dir"
rm "$openslr_dir/rirs_noises.zip"
