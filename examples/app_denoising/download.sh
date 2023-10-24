#!/usr/bin/bash

BLOB_NAMES=(
    clean_fullband/datasets_fullband.clean_fullband.french_speech_000_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_001_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_002_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_003_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_004_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_005_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_006_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_007_NA_NA.tar.bz2
    # clean_fullband/datasets_fullband.clean_fullband.french_speech_008_NA_NA.tar.bz2
)
AZURE_URL="https://dns4public.blob.core.windows.net/dns4archive/datasets_fullband"

OUTPUT_PATH="./datasets_fullband"

mkdir -p $OUTPUT_PATH/{clean_fullband,noise_fullband}

for BLOB in ${BLOB_NAMES[@]}
do
    URL="$AZURE_URL/$BLOB"
    echo "Download: $BLOB"
    curl "$URL" | tar -C "$OUTPUT_PATH" -f - -x -j
done
