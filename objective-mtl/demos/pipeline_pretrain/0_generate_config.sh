#!/bin/bash

data_type=$1
data_name=$2
template_file=$3
output_file=$4

cat $template_file | sed "s/data_type/$data_type/g" | sed "s/data_name/$data_name/g" > $output_file
chmod 777 $output_file
