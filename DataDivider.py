## 시계열 데이터이고, 입력의 형태가 특정 길이(window size)의 sequence 데이터 이므로 shuffle은 사용하지 않음.
## Normal 데이터는 학습데이터, 파라미터 설정데이터, 검증용데이터, 실험용데이터의 비율을 7:1:1:1 로 나누어서 사용합니다.

def divide(data):
    interval_n = int(len(data)/10)
    data_df1 = data.iloc[0:interval_n*7]
    data_df2 = data.iloc[interval_n*7:interval_n*8]
    data_df3 = data.iloc[interval_n*8:interval_n*9]
    data_df4 = data.iloc[interval_n*9:]

    ## 데이터 정규화를 위하여 분산 및 평균 추출
    mean_df = data_df1.mean()
    std_df = data_df1.std()

    return data_df1, data_df2, data_df3, data_df4, mean_df, std_df
