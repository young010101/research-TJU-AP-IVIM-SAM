#!/bin/bash

# Get the current date in YYYYMMDD format
CURRENT_DATE="20240918_190519"

# Specify the base folder where the NIfTI output is stored
BASE_NIFTI_DIR="/data/users/cyang/acute_pancreatitis/unprocess/nii"

# Combine the base directory with the current date
TAG="ivimap_9_patients"
NIFTI_DIR="${BASE_NIFTI_DIR}/${TAG}/${CURRENT_DATE}"

# Change to the NIfTI directory
cd "$NIFTI_DIR" || { echo "NIFTI directory not found!"; exit 1; }

# Base directory for the bval files
OUTPUT_BASE_DIR="${BASE_NIFTI_DIR}/../bval/${TAG}/${CURRENT_DATE}"
mkdir -p "$OUTPUT_BASE_DIR"
# Clear $OUTPUT_BASE_DIR
rm -rf "$OUTPUT_BASE_DIR"/*

# Loop through each patient's directory
for patient_id in *; do
    if [ -d "$patient_id" ]; then
        cd "$patient_id" || continue
        echo "Processing directory: $patient_id"

        base_filename_1="${patient_id%%-*}"
        echo "Base filename: $base_filename_1"

    # Loop through each .bval file in the patient directory
    for file in *.bval; do
        # Check if any .bval files exist
        if [ ! -e "$file" ]; then
                echo "No .bval files found in $patient_id"
                continue
            fi

            echo "Processing file: $file"

            # Extract the base file name without extension
            base_filename="${file%.*}"
            echo "Base filename: $base_filename"

            # IFS='.' read -r -a array <<< "$file"
            # echo "IVIM filename: ${array[0]}"

            cat "$file"

            # Create the patient's directory in the output location
            patient_output_dir="${OUTPUT_BASE_DIR}/${base_filename_1}"
            mkdir -p "$patient_output_dir"

            # Copy the .bval file to the patient's directory
            cp "$file" "${patient_output_dir}/"

            # Copy the corresponding .nii.gz file
            nii_file="${base_filename}.nii.gz"
            if [ -e "$nii_file" ]; then
                cp "$nii_file" "${patient_output_dir}/"
            else
                echo "Warning: Corresponding .nii.gz file not found for $file"
            fi
        done
        echo "-----------------------------------"
        cd ..
    fi
done
