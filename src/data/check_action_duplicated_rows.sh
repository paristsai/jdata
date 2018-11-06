#!/bin/bash

# spent total 19:09.49 sec but cannot finish

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../data/ && pwd)"
INTERIM_DIR="$DATA_DIR/interim"

sort "$INTERIM_DIR/All_Action.csv" | uniq -c | sort -nr | awk '{if($1 != 1) {sum += $1}} END {print sum}'