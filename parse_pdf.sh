#!/bin/bash

# we need the path to the jar file parsing the pdfs from science parse
path=$1

echo Path to science parse jar file: $path

echo Start parsing pdfs ...
cd data/pdfs/
java -Xmx6g -jar ../../$path ./ -o ../json/
