#!/bin/bash

# Specify the folder where the DICOM files are located
#DICOM_DIR="/path/to/dicom/folder"
DICOM_DIR="/data/users/cyang/acute_pancreatitis/unprocess/dcm/
"

# Specify the folder where the NIfTI output should be stored
NIFTI_DIR="/data/users/cyang/acute_pancreatitis/unprocess/nii_tmp"

# Check if dcm2niix is installed
if ! command -v dcm2niix &> /dev/null; then
    echo "dcm2niix could not be found, please install it before running this script."
    exit 1
fi

# Start the timer
start_time=$(date +%s)

# Change to the DICOM directory
cd "$DICOM_DIR"

# Loop through each patient's DICOM folder
for patient_id in *; do
    # Check if it is a directory
    if [ -d "$patient_id" ]; then
        # Create the output directory for the patient
        output_dir="$NIFTI_DIR/$patient_id"
        mkdir -p "$output_dir"

        # Convert all DICOM files for the patient
        dcm2niix -b y -f "%s_%p_%s" -z y "$patient_id" "$output_dir"
    fi
done

# End the timer
end_time=$(date +%s)

# Calculate the elapsed time in seconds
elapsed_time=$((end_time - start_time))

# Print the conversion completion message and the elapsed time
echo "Conversion completed in $elapsed_time seconds."
