#!/bin/bash

# Specify the folder where the DICOM files are located
DICOM_DIR="/data/users/cyang/acute_pancreatitis/unprocess/dcm/ivimap/ivimap/apivim"

# Get the current date in YYYYMMDD format
CURRENT_DATE=$(date +%Y%m%d_%H%M%S)

# Specify the base folder where the NIfTI output should be stored
BASE_NIFTI_DIR="/data/users/cyang/acute_pancreatitis/unprocess/nii"

# Combine the base directory with the current date
TAG="ivimap_9_patients"
NIFTI_DIR="${BASE_NIFTI_DIR}/${TAG}/${CURRENT_DATE}"

LOG_FILE="conversion.log"

# Check if dcm2niix is installed
if ! command -v dcm2niix &> /dev/null; then
    echo "dcm2niix could not be found, please install it before running this script."
    exit 1
fi

# Start the timer and log
start_time=$(date +%s)
echo "Conversion started at $(date)" > "$LOG_FILE"

# Change to the DICOM directory
cd "$DICOM_DIR" || { echo "DICOM directory not found!"; exit 1; }

# Loop through each patient's DICOM folder
for patient_id in *; do
    if [ -d "$patient_id" ]; then
        dicom_path="$patient_id"
        output_dir="$NIFTI_DIR/$patient_id"
        mkdir -p "$output_dir"

        if [ -d "$dicom_path" ]; then
            echo "Converting DICOM files for $patient_id..."
            if dcm2niix -b y -f "%p_%s" -o "$output_dir" -z y "$dicom_path"; then
                echo "$patient_id: Conversion successful." >> "$LOG_FILE"
            else
                echo "$patient_id: Conversion failed." >> "$LOG_FILE"
            fi
        else
            echo "$patient_id: $ST0_DIR not found. Skipping." >> "$LOG_FILE"
        fi
    fi
done

# End the timer and log
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "Conversion completed in $elapsed_time seconds."
echo "Conversion completed at $(date)" >> "$LOG_FILE"
