#!/bin/sh
echo "RUNNING SCRIPT"

# in xformer folder
TAG=$(git describe --tags --abbrev=0)
GIT_MAJOR=$(echo ${TAG} | cut -d. -f1 | sed 's/v//')
GIT_MINOR=$(echo ${TAG} | cut -d. -f2)
GIT_PATCH=$(echo ${TAG} | cut -d. -f3)

echo $GIT_MAJOR

MINOR2=$(grep 'minor' Version.h | awk '{print $4}' | sed 's/;//')
echo $MINOR2

# if [ "$MINOR1" != "$MINOR2" ] || [ "$MINOR1" != "$MINOR2" ]
# then exit 1
# fi

pwd
LIB_NN_VERSION_H="../../external/lib_nn/lib_nn/api/version.h"

function myfunc()
{
    local type=$1
    local filename=$2
    MINOR=$(grep $type $filename | awk '{print $6}' | sed 's/;//')
    echo "$MINOR"
}

result=$(myfunc "major" $LIB_NN_VERSION_H)
echo $result
result=$(myfunc "minor" $LIB_NN_VERSION_H)
echo $result
result=$(myfunc "patch" $LIB_NN_VERSION_H)
echo $result

function myfunc2()
{
    local filename=$1
    MAJOR=$(grep 'major' $filename | awk '{print $6}' | sed 's/;//')
    MINOR=$(grep 'minor' $filename | awk '{print $6}' | sed 's/;//')
    PATCH=$(grep 'patch' $filename | awk '{print $6}' | sed 's/;//')
    echo "$MAJOR.$MINOR.$PATCH"
}

LIB_TFLITE_MICRO_VERSION_H="../../external/lib_tflite_micro/lib_tflite_micro/api/version.h"
result=$(myfunc2 $LIB_TFLITE_MICRO_VERSION_H)
echo $result

LIB_TFLITE_MICRO_MODULE_BUILD_INFO="../../external/lib_tflite_micro/lib_tflite_micro/module_build_info"
MINOR=$(grep 'VERSION' $LIB_TFLITE_MICRO_MODULE_BUILD_INFO | awk '{print $3}')

echo $MINOR

if [ "$MINOR" != "$result" ]
then exit 1
fi

exit 0
