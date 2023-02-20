#!/bin/bash

CUR_DIR=$(pwd)

cd ../../external/lib_tflite_micro/
if ! ./version_check.sh; then
    exit 1
fi

cd $CUR_DIR
printf "\nRunning version check for xformer..."

# in xformer folder
TAG=$(git describe --tags --abbrev=0)
GIT_VERSION=$(printf ${TAG} | sed 's/v//')

printf "\nGit version = "$GIT_VERSION

function get_version()
{
    local filename=$1
    MAJOR=$(grep 'major' $filename | awk '{print $4}' | sed 's/;//')
    MINOR=$(grep 'minor' $filename | awk '{print $4}' | sed 's/;//')
    PATCH=$(grep 'patch' $filename | awk '{print $4}' | sed 's/;//')
    printf "$MAJOR.$MINOR.$PATCH"
}

VERSION_H="Version.h"

VERSION_H_STR=$(get_version $VERSION_H)
printf "\nVersion header = "$VERSION_H_STR

if [ "$GIT_VERSION" != "$VERSION_H_STR" ]
then printf "\nVersion mismatch!" && exit 1
fi

printf "\n"
exit 0
