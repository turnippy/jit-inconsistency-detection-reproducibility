import pandas as pd
import numpy as np
import os
import re

set_path = lambda x,y: f'{os.path.dirname(os.path.abspath(__file__))}{x}/{y}'
encode = lambda x:x.encode('utf-8').decode('unicode_escape')

def extractData(data):
    patten = r'(\d+)\,( \w+)*\,( \w+)*\,( \w+)*'
    first_line = [data.strip() for data in re.findall(patten,repr(data))[0]]
    patten = r'\\n(.*)\,( \d+)*,( \d+)*\\n\d+\\n.*?\\n\d+\\n'
    second_line = [data.strip() for data in re.findall(patten,repr(data))[0]]
    second_line = second_line[0].split("\\n")[-1:] + second_line[1:]
    patten = r'\\n(\d+)\\n(.*?)\\n(\d+)\\n(.*?)###'
    result = re.findall(patten,repr(data))
    b_result = [b for b in result[0]]
    merge = first_line + second_line + b_result

    return merge

def convert(name):
    path = f"/data/{name}"
    files = os.listdir(os.path.dirname(os.path.abspath(__file__))+path)
    
    for file in files: 
        if 'Coherence' in file:
            Coherence_Data_path = set_path(path,file)
        else:
            Raw_Data_path = set_path(path,file)

    with open(Coherence_Data_path,"r") as file:
        data = file.read().split("\n")[:-1]
        data_list = [[d.strip() for d in da.split(",")] for da in data]
    df = pd.DataFrame(data_list,columns=['method_id', 'coherence'])
    df.to_csv(set_path('/convert_data',f'{name}_Coherence_data.csv'),index=False)
    
    result = list()
    with open(Raw_Data_path,"r") as file:
        data =file.read().split("###")[:-1]
        data_end = [da+"###" for da in data]
        for de in data_end: result.append(extractData(de))
    df = pd.DataFrame(result,columns=['method_id','method_name','class_name','software_system','filepath','start_line','end_line','Length_of_Head_Comment','Head_Comment','Length_of_the_Implementation','Method_Implementation'])
    df.to_csv(set_path('/convert_data',f'{name}_Raw_data.csv'),index=False)

if __name__ == "__main__":
    names = ['CoffeeMaker_Web','ifreechart_060','jfreechart_071','jhotdraw7']
    for name in names:
        convert(name)