import os
from ivim_analysis.n_patients import readPatientsInfo
import unittest

class TestReadNPatientsInfo(unittest.TestCase):
    def test_example(self):
        # test the function readPatientsInfo
        patients_info_file = 'ivim_analysis/test/4patients.txt'
        zhaog_path = '/data/users/cyang/acute_pancreatitis/unprocess/ivim'
        n_analyses = readPatientsInfo(patients_info_file, zhaog_path)
        self.assertEqual(len(n_analyses), 4)
        self.assertEqual(n_analyses[0].patient_id, "DongWeilong")
        self.assertEqual(n_analyses[0].nii_path, os.path.join(zhaog_path,"DongWeilong.nii.gz"))
        self.assertEqual(n_analyses[0].x_roi, 129)
        self.assertEqual(n_analyses[0].y_roi, 156)
        self.assertEqual(n_analyses[0].rad, 4)
    
    def test_run_ivim(self):
        patients_info_file = 'ivim_analysis/test/4patients.txt'
        zhaog_path = '/data/users/cyang/acute_pancreatitis/unprocess/ivim'
        n_analyses = readPatientsInfo(patients_info_file, zhaog_path)
        # n_analyses[0].run_analysis()

if __name__ == '__main__':
    unittest.main()