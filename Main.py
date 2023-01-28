import torch
import easydict
from TagDataSet import TagDataset
import DataLoader
import DataPreprocessor
import DataDivider
from Model import LSTMAutoEncoder
import Train
import numpy as np
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

data = DataLoader.load()
DataPreprocessor(data)
data_df1, data_df2, data_df3, data_df4, mean_df, std_df = DataDivider('data.csv')

## 설정 폴더
args = easydict.EasyDict({
    "batch_size": 128, ## 배치 사이즈 설정
    "device": torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), ## GPU 사용 여부 설정
    "input_size": 3, ## 입력 차원 설정
    "latent_size": 1, ## Hidden 차원 설정
    "output_size": 3, ## 출력 차원 설정
    "window_size" : 3, ## sequence Lenght
    "num_layers": 2,     ## LSTM layer 갯수 설정
    "learning_rate" : 0.001, ## learning rate 설정
    "max_iter" : 100000, ## 총 반복 횟수 설정
    'early_stop' : True,  ## valid loss가 작아지지 않으면 early stop 조건 설정
})


## 데이터셋으로 변환
normal_dataset1 = TagDataset(df=data_df1, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
normal_dataset2 = TagDataset(df=data_df2, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
normal_dataset3 = TagDataset(df=data_df3, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
normal_dataset4 = TagDataset(df=data_df4, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)

## Data Loader 형태로 변환
train_loader = torch.utils.data.DataLoader(
                 dataset=normal_dataset1,
                 batch_size=args.batch_size,
                 shuffle=False)
valid_loader = torch.utils.data.DataLoader(
                dataset=normal_dataset2,
                batch_size=args.batch_size,
                shuffle=False)

## 모델 생성
model = LSTMAutoEncoder(input_dim=args.input_size, latent_dim=args.latent_size, window_size=args.window_size, num_layers=args.num_layers)
model.to(args.device)

## 학습하기
model = Train.run(args, model, train_loader, valid_loader)
torch.save(model.state_dict(), '/result/model8.pth')
torch.save(model, '/result/model8.pt')

## Loss를 구하기
loss_list = Train.get_loss_list(args, model, valid_loader)

## Reconstruction Error의 평균과 Covarinace 계산
mean = np.mean(loss_list, axis=0)
std = np.cov(loss_list.T)


## Anomaly Score
class Anomaly_Calculator:
    def __init__(self, mean: np.array, std: np.array):
        assert mean.shape[0] == std.shape[0] and mean.shape[0] == std.shape[1], '평균과 분산의 차원이 똑같아야 합니다.'
        self.mean = mean
        self.std = std

    def __call__(self, recons_error: np.array):
        x = (recons_error - self.mean)
        return np.matmul(np.matmul(x, self.std), x.T)


## 비정상 점수 계산기
anomaly_calculator = Anomaly_Calculator(mean, std)

## Threshold 찾기
anomaly_scores = []
for temp_loss in tqdm(loss_list):
    temp_score = anomaly_calculator(temp_loss)
    anomaly_scores.append(temp_score)

## 정상구간에서 비정상 점수 분포
print("평균[{}], 중간[{}], 최소[{}], 최대[{}]".format(np.mean(anomaly_scores), np.median(anomaly_scores), np.min(anomaly_scores), np.max(anomaly_scores)))

anomaly_calculator = Anomaly_Calculator(mean, std)

## 전체 데이터 불러오기
total_dataset = TagDataset(df=data, input_size=args.input_size, window_size=args.window_size, mean_df=mean_df, std_df=std_df)
total_dataloader = torch.utils.data.DataLoader(dataset=total_dataset,batch_size=args.batch_size,shuffle=False)

## Reconstruction Loss를 계산하기
total_loss = Train.get_loss_list(args, model, total_dataloader)

## 이상치 점수 계산하기
anomaly_scores = []
for temp_loss in tqdm(total_loss):
    temp_score = anomaly_calculator(temp_loss)
    anomaly_scores.append(temp_score)

visualization_df = total_dataset.df
visualization_df['score'] = anomaly_scores

## 시각화 하기
fig = plt.figure(figsize=(16, 6))
ax=fig.add_subplot(111)

visualization_df['score'].plot(ax=ax)
ax.legend(['abnormal score'], loc='upper right')
