#!/usr/bin/env bash
#
# Copyright (c) 2020, XMOS Ltd, All rights reserved

echo_and_run() { echo "$*" ; "$@" ; }

XE=$1
TRACE_TO=$2

if [[ $# -ne 2 ]]; then
    ARGS=$3
    echo_and_run xsim --trace --enable-fnop-tracing --args ${XE} ${ARGS} --trace-to ${TRACE_TO}
else
    echo_and_run xsim --trace --enable-fnop-tracing --trace-to $TRACE_TO $XE
fi

echo_and_run python process_trace.py -t ${TRACE_TO} 