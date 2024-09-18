import os
from pypinyin import pinyin, Style


def extract_patient_and_b_values(ivim_path):
    filenames = os.listdir(ivim_path)
    patient = [file.split("_")[0] for file in filenames if file.endswith(".nii.gz")]
    num_b = [
        file.split("_")[1].split(".")[0]
        for file in filenames
        if file.endswith(".nii.gz")
    ]
    patient_num_b_dict = dict(zip(patient, num_b))
    return patient_num_b_dict


def chinese_to_pinyin(name):
    pinyin_name = pinyin(name, style=Style.NORMAL)
    flattened = [item[0].capitalize() for item in pinyin_name]
    result = "".join(flattened)
    return result
