import yaml
from nptdms import TdmsFile

def load_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def load_tdms_file(file_path):
    tdms_file = TdmsFile.read(file_path)
    df = tdms_file.as_dataframe()
    return df