import os
import pandas as pd
from typing import List

root = '/data/'


def get_data_names(date: str):
    return sorted([data for data in os.listdir(root) if date in data])


def divide_machine(names: str):
    machine1 = [name for name in names if 'tem' in name if 'ai0' in name] + [name for name in names if 'vib' in name if 'ai0' in name or 'ai1' in name]
    machine2 = [name for name in names if name not in machine1]
    return machine1, machine2


def concat_data(data_names: List):
    data_list = []

    for data_name in data_names:
        data_path = os.path.join(root, data_name)
        data_name_full = data_name.split('.')[0]

        if 'tem' in data_name_full:
            data = pd.read_csv(data_path, names=['time', 'temp'])
        elif 'ai0' in data_name_full or 'ai2' in data_name_full:
            data = pd.read_csv(data_path, names=['time', 'left'])
        else:
            data = pd.read_csv(data_path, names=['time', 'right'])

        del data['time']
        data_list.append(data)

    concated = pd.concat(data_list, axis=1)

    return concated


#date_list 에 사용할 데이터들의 날자들을 입력하면, 데이터 타입과 규칙을 이용하여, 데이터를 생성
def load():
    date_list = ['20221103', '20221102', '20221114']
    names = [get_data_names(date) for date in date_list]
    divided = [divide_machine(name) for name in names]
    concated = [concat_data(nameList) for tuples in divided for nameList in tuples]
    data = pd.concat(concated, axis=0)
    return data
