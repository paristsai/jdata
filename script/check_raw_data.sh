#!/bin/bash

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../data/ && pwd)"
RAW_DIR="$DATA_DIR/raw"

files=("JData_Action_201602.csv" "JData_Action_201603.csv" "JData_Action_201604.csv" "JData_User.csv" "JData_Product.csv" "JData_Comment.csv")
for i in "${files[@]}"
do
    if [ ! -f $RAW_DIR"/"$i ]
    then
        echo "Missing the file: "$i
        echo "Please download the data from the link: https://pan.baidu.com/s/1i4QC8lv"
        exit 1
    fi
done