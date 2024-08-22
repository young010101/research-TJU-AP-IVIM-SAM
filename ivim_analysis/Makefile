# Makefile for ivim_analysis project

# Define the Python interpreter
PY_INTERP = python

# Define the main Python script
MAIN_SCRIPT = main.py

# Define the directory where data is stored
DATA_DIR = data/

# Define the output directory for generated files
OUTPUT_DIR = output/
NII_TMP_DIR = /data/users/cyang/acute_pancreatitis/unprocess/nii_tmp/

# Rule to run the main Python script
run:
	$(PY_INTERP) $(MAIN_SCRIPT)

# Rule to clean up generated files and directories
clean:
	rm -rf $(OUTPUT_DIR)*
	
# Rule to create the output directory if it does not exist
prepare:	
	mkdir -p $(OUTPUT_DIR)

# Rule to convert dcm2nii
nii:
	./convert_dcm2nii.sh

# Rule to clean up nii_tmp files and dir in /data/users/cyang/acute_pancreatitis/unprocess/nii_tmp/data
clean_nii_tmp:
	rm -rf $(NII_TMP_DIR)

# Default target
all: prepare run clean

# Include phony target to indicate that clean is not a file
.PHONY: clean prepare all