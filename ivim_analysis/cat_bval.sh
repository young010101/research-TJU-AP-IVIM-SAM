#!/bin/bash

# Get the current date in YYYYMMDD format
CURRENT_DATE="20240918_190519"

# Specify the base folder where the NIfTI output should be stored
BASE_NIFTI_DIR="/data/users/cyang/acute_pancreatitis/unprocess/nii"

# Combine the base directory with the current date
TAG="ivimap_9_patients"
NIFTI_DIR="${BASE_NIFTI_DIR}/${TAG}/${CURRENT_DATE}"

# Change to the DICOM directory
cd "$NIFTI_DIR" || { echo "NIFTI directory not found!"; exit 1; }

# TODO: Get patient name $patient_name from the file name and mkdir in ../bval
# Example:
#   Input: 刘伯林-MR202403270041-AB+MRCP-2024-03-27-20240912112153710
#   Output: 刘伯林

# Loop through each patient's DICOM folder
for patient_id in *; do
    if [ -d "$patient_id" ]; then
        cd "$patient_id" || continue
        echo "Contents of $file: in $patient_id"
        for file in *.bval; do
            # echo file name
            echo "file name: $file"
            # split the file name by .
            IFS='.' read -r -a array <<< "$file"
            # get the first element of the array
            echo "first element: ${array[0]}"

            cat "$file"

            # TODO: cp ${file} and ${array[0]}.nii.gz to ../bval/${patient_name}
        done
        echo "-----------------------------------"
        cd ..
    fi
done
