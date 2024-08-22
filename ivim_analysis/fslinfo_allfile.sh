#!/bin/bash

# Define the directory path
#dir="/data/users/cyang/acute_pancreatitis/unprocess/nii_tmp_2/PA1"
dir="/data/users/cyang/acute_pancreatitis/unprocess/nii/pantient2/"

# Use the find command to search for files ending with '.nii.gz' in the specified directory
#find "$dir" -type f -name "*.nii.gz" -exec echo "Found .nii.gz file: {}" \; -exec fslinfo {} \;

cd $dir

# Loop through each .nii.gz file in the directory
for file in *.nii.gz
do
    echo "File: $file"
    fslinfo "$file" | grep "dim4" |grep 10
done