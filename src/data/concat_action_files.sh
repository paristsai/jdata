#!/bin/bash

set -e

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../../data/ && pwd)"
RAW_DIR="$DATA_DIR/raw"
INTERIM_DIR="$DATA_DIR/interim"
ALL_ACTION_FILE="$INTERIM_DIR/All_Action.csv"

FILE_LIST=$(find $RAW_DIR -type f -name "JData_Action*.csv")
printf -- 'Find action data files:\n'
echo "$FILE_LIST"
printf -- '\nStart concating...'
echo "$FILE_LIST" | head -1 | xargs head -1 >$ALL_ACTION_FILE

# append rows with no header
echo $FILE_LIST | xargs -n 1 sed "1d" >>$ALL_ACTION_FILE
# for filename in $FILE_LIST; do sed "1d" $filename >>$ALL_ACTION_FILE; done

printf -- ' \033[32mDONE\033[0m\n'
printf -- '\nResult file:\n%s\n' "$ALL_ACTION_FILE"

exit 0
