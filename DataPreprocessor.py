def preprocess(data):
    data = data.dropna()
    data.to_csv('data,csv')
