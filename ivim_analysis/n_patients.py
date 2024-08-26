import os 
from ivim_analysis.IVIMAnalysis import IVIMAnalysis

def readPatientsInfo(patients_info_file, path=None):
    with open(patients_info_file, 'r') as f:
        # Read and print the next int
        num = int(f.readline())
        f.readline()
        
        # patient_id, nii_name, x_roi, y_roi, rad
        info_patients = []
        for i in range(num):
            info_patients.append(f.readline().split())
            
    n_analyses = []
    for i in range(num):
        patient_id, nii_name, x_roi, y_roi, rad = info_patients[i]
        example_filename = os.path.join(path, nii_name)
        analysis = IVIMAnalysis(example_filename, int(x_roi), int(y_roi), int(rad), patient_id)
        n_analyses.append(analysis)

    return n_analyses