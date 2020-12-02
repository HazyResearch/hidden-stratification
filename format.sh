#!/bin/bash

if [ $# -eq 0 ]; then
    cd $(dirname "${BASH_SOURCE[0]}")
    for f in $(git ls-tree -r $(git rev-parse --abbrev-ref HEAD) --name-only | grep -E "*.py$");
        do yapf -ir -vv $f
    done
else
    yapf -ir -vv $1
fi
