import json


def get_config(json_file: str):
    """read path from json file"""
    with open(json_file) as f:
        data = json.load(f)
        patients_info_file = data["patients_info_file"]
        zhaog_path = data["zhaog_path"]
        output_path = data["output_path"]

    return patients_info_file, zhaog_path, output_path
