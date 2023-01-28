from torch.utils.data import Dataset
import numpy as np
import torch


## 데이터를 불러올 때 index로 불러오기
def make_data_idx(dates, window_size=1):
    input_idx = []
    for idx in range(window_size-1, len(dates)):
        input_idx.append(list(range(idx - window_size+1, idx+1)))
    return input_idx


## Dataset을 상속받아 데이터를 구성
class TagDataset(Dataset):
    def __init__(self, input_size, df, mean_df=None, std_df=None, window_size=1):
        ## 변수 갯수
        self.input_size = input_size

        ## 복원할 sequence 길이
        self.window_size = window_size

        ## Summary용 데이터 Deep copy
        original_df = df.copy()

        ## 정규화
        if mean_df is not None and std_df is not None:
            sensor_columns = [item for item in df.columns]
            df[sensor_columns] = (df[sensor_columns] - mean_df) / std_df

        ## 연속한 index를 기준으로 학습에 사용합니다.
        index_list = df.index.to_list()
        self.input_ids = make_data_idx(index_list, window_size=window_size)

        ## sensor 데이터만 사용하여 reconstruct에 활용
        self.selected_column = [item for item in df.columns][:input_size]
        self.var_data = torch.tensor(df[self.selected_column].values.astype(np.float), dtype=torch.float)

        ## Summary 용
        self.df = original_df.iloc[np.array(self.input_ids)[:, -1]]

    ## Dataset은 반드시 __len__ 함수를 만들어줘야함(데이터 길이)
    def __len__(self):
        return len(self.input_ids)

    ## Dataset은 반드시 __getitem__ 함수를 만들어줘야함
    ## torch 모듈은 __getitem__ 을 호출하여 학습할 데이터를 불러옴.
    def __getitem__(self, item):
        temp_input_ids = self.input_ids[item]
        input_values = self.var_data[temp_input_ids]
        return input_values
