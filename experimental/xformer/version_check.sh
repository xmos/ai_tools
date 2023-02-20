#!/bin/sh

CUR_DIR=$(pwd)

cd ../../external/lib_tflite_micro/
if ! ./version_check.sh; then
    exit 1
fi

cd $CUR_DIR
echo "\nRunning version check for xformer..."

# in xformer folder
TAG=$(git describe --tags --abbrev=0)
GIT_VERSION=$(echo ${TAG} | sed 's/v//')

echo "Git version = "$GIT_VERSION

function get_version()
{
    local filename=$1
    MAJOR=$(grep 'major' $filename | awk '{print $4}' | sed 's/;//')
    MINOR=$(grep 'minor' $filename | awk '{print $4}' | sed 's/;//')
    PATCH=$(grep 'patch' $filename | awk '{print $4}' | sed 's/;//')
    echo "$MAJOR.$MINOR.$PATCH"
}

VERSION_H="Version.h"

VERSION_H_STR=$(get_version $VERSION_H)
echo "Version header = "$VERSION_H_STR

if [ "$GIT_VERSION" != "$VERSION_H_STR" ]
then echo "Version mismatch!" && exit 1
fi

exit 0
