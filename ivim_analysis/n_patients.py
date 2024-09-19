import os
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

        n_analyses = {}
        for i in range(num):
            _, nii_name, pancreas_slice, x_roi, y_roi, rad = info_patients[i]
            patient_id = nii_name.split("_")[0]
            num_b = nii_name.split("_")[1].split(".")[0]
            example_filename = os.path.join(path, nii_name)
            analysis = IVIMAnalysis(
                example_filename,
                int(pancreas_slice),
                int(num_b),
                int(x_roi),
                int(y_roi),
                int(rad),
                patient_id,
            )
            n_analyses[patient_id] = analysis

        return n_analyses

    def runIVIMAnalysis(n_analyses):
        for analysis in n_analyses:
            analysis.fit_ivim_model(analysis.bvals, analysis.bvecs)
