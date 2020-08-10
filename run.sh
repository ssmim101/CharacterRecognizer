#!/bin/bash -e
BASEPATH=$(cd "$(dirname "$0")" && pwd)
julia $BASEPATH/src/main.jl $BASEPATH/res
