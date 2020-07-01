#!/bin/bash

cd $(dirname "${BASH_SOURCE[0]}")
for f in $(git ls-tree -r $(git rev-parse --abbrev-ref HEAD) --name-only | grep -E "*.py$");
    do yapf -ir -vv $f
done
