import os
import json

path = "/home/lofowl/Desktop/data"
data_elements = os.listdir(path)
param_path = f'{path}/{data_elements[0]}'

def ParamDataSet(dataset,*kwgs) -> list:
    
    param_test = f'{param_path}/{dataset}.json'
    with open(param_test,"r") as files: 
        param_test = json.loads(files.read())
    result = {kwg:[] for kwg in kwgs}
    for param in param_test:
        print(param.keys())
        for kwg in kwgs:
            result[kwg] = result[kwg] + [param[kwg]]
    return list(result.values())

if __name__ == "__main__":
    data_elements = os.listdir(path)
    
    param_path = f'{path}/{data_elements[0]}'
    result = ParamDataSet('train',"old_comment_raw","old_code_raw")
    
    train = ParamDataSet('test',"old_comment_raw","old_code_raw")
    
    