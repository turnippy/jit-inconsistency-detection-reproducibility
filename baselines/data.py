import os
import json

path = "./data"
data_elements = os.listdir(path)
param_path = f'{path}/{data_elements[0]}'

def ParamTest(*kwgs) -> list:
    
    param_test = f'{param_path}/test.json'
    with open(param_test,"r") as files: 
        param_test = json.loads(files.read())
    result = {kwg:[] for kwg in kwgs}
    for param in param_test:
        for kwg in kwgs:
            result[kwg] = result[kwg] + [param[kwg]]
    return result.values()

if __name__ == "__main__":
    data_elements = os.listdir(path)
    
    param_path = f'{path}/{data_elements[0]}'
    result = ParamTest("old_comment_raw","old_code_raw")
    
    
    