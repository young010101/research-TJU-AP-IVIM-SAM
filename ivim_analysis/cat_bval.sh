#!/bin/bash

# Get the current date in YYYYMMDD format
CURRENT_DATE=$(date +%Y%m%d_%H%M%S)
FIX_CURRENT_DATE="20240918_190519"

# Specify the base folder where the NIfTI output is stored
BASE_NIFTI_DIR="/data/users/cyang/acute_pancreatitis/unprocess/nii"

# Combine the base directory with the current date
TAG="ivimap_9_patients"
NIFTI_DIR="${BASE_NIFTI_DIR}/${TAG}/${FIX_CURRENT_DATE}"

# Change to the NIfTI directory
cd "$NIFTI_DIR" || { echo "NIFTI directory not found!"; exit 1; }

# Base directory for the bval files
FLAG_STATIC_DIR=true
if [ "$FLAG_STATIC_DIR" = true ]; then
    OUTPUT_BASE_DIR="${BASE_NIFTI_DIR}/../bval/${TAG}/static"
else
    OUTPUT_BASE_DIR="${BASE_NIFTI_DIR}/../bval/${TAG}/${FIX_CURRENT_DATE}/${CURRENT_DATE}"
fi

mkdir -p "$OUTPUT_BASE_DIR"

BASE_LOG_FILE="cat_bval.log"
LOG_FILE="${OUTPUT_BASE_DIR}/${BASE_LOG_FILE}"

# Clear $OUTPUT_BASE_DIR
rm -rf "$OUTPUT_BASE_DIR"/*
echo "Clearing $OUTPUT_BASE_DIR" >> "$LOG_FILE"

# Loop through each patient's directory
for patient_id in *; do
    if [ -d "$patient_id" ]; then
        cd "$patient_id" || continue
        echo "Processing directory: $patient_id"

        base_filename_1="${patient_id%%-*}"
        echo "Base filename: $base_filename_1" >> "$LOG_FILE"
        all_files_less_than_10=true

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
            echo "Base filename: $base_filename" >> "$LOG_FILE"

            # IFS='.' read -r -a array <<< "$file"
            # echo "IVIM filename: ${array[0]}"

            cat "$file" >> "$LOG_FILE"

            # Split the 1st line of the .bval file by space
            # Read the first line of the .bval file
            read -r first_line < "$file"

            # Split the first line into an array
            IFS=' ' read -r -a bval_array <<< "$first_line"
            
            # Output the array size
            array_size="${#bval_array[@]}"
            echo "# of b: $array_size" >> "$LOG_FILE"

            # Check if array size >= 10
            if [ "$array_size" -ge 10 ]; then
                massage="[v]Array size $array_size is greater than or equal to 10 for file $file in $base_filename_1"
                echo $massage >> "$LOG_FILE"
                echo $massage
                # Since we found a file with array size >=10, set the flag to false
                all_files_less_than_10=false
            fi

            # Output the values
            # echo "Values in the .bval file:"
            # for value in "${bval_array[@]}"; do
            #     echo "$value"
            # done

            # while IFS= read -r line; do
            #     echo "$line"
            # done < "$file"

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

        # After processing all files in patient directory
        if [ "$all_files_less_than_10" = true ]; then
            massage="[x]All files in $base_filename_1 have array sizes less than 10."
            echo $massage >> "$LOG_FILE"
            echo $massage
        fi

        echo "-----------------------------------" >> "$LOG_FILE"
        echo "-----------------------------------"
        cd ..
    fi
done
