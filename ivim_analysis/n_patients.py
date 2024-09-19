import os
import json
from ivim_analysis.IVIMAnalysis import IVIMAnalysis


class NPatients:
    """
    Given a file with the information of the patients,
    this class reads the information and creates a list of IVIMAnalysis objects.
    """

    def readPatientsInfo(patients_info_file, path=None):
        with open(patients_info_file, "r") as f:
            # Read and print the next int
            num = int(f.readline())
            f.readline()

            # patient_id, nii_name, x_roi, y_roi, rad
            info_patients = []
            for i in range(num):
                info_patients.append(f.readline().split())

        with open("../config/circles.json", "r") as f:
            circ = json.load(f)

        n_analyses = {}

        # Get the b-values for each patient
        dir_bvals = NPatients.dir_b_values(path)

        for i in range(num):
            _, nii_name, pancreas_slice, x_roi, y_roi, rad = info_patients[i]
            patient_id = nii_name.split("_")[0]
            num_b = nii_name.split("_")[1].split(".")[0]
            example_filename = os.path.join(path, nii_name)

            # Create an IVIMAnalysis object for each patient
            analysis = IVIMAnalysis(
                example_filename,
                int(pancreas_slice),
                int(num_b),
                int(x_roi),
                int(y_roi),
                int(rad),
                patient_id,
            )

            # Add the roi circles to the analysis
            analysis.circles = circ[patient_id]

            # Add b-values to the analysis
            analysis.bvals = dir_bvals[patient_id]
            vec = [0, 0, 1]
            analysis.bvecs = [vec] * len(analysis.bvals)

            n_analyses[patient_id] = analysis

        return n_analyses

    def runIVIMAnalysis(n_analyses):
        for analysis in n_analyses:
            analysis.fit_ivim_model(analysis.bvals, analysis.bvecs)

    def dir_b_values(path):
        dir_bvals = {}
        b_val_path = os.path.join(path, "patient.bval")
        with open(b_val_path, "r") as f:
            b_val = f.read().split("\n")
            # remove the last empty line
            b_val = b_val[:-1]
            num_patients = len(b_val)
            print("# of patients: ", num_patients)
            for i in range(num_patients):
                # tmp_b_val = [int(i) for i in b_val[0] if i.isdigit()]  # Filter out non-numeric values
                i_name_bval = b_val[i].split()
                i_name = i_name_bval[0]
                i_b_val = [
                    int(i) for i in i_name_bval if i.isdigit()
                ]  # Filter out non-numeric values
                dir_bvals[i_name] = i_b_val

        return dir_bvals
