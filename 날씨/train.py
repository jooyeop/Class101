import numpy as np
import tensorflow as tf

FILE_NAME = '서초1동_기온_202006_202105.csv'

def load_data():
    data = []
    with open(f'./data/{FILE_NAME}') as f: # CSV파일을 파이썬에서 읽을 수 있게 load하는 코드
        header = f.readline() # 파일의 첫번째 줄에는 열에 대한 정보인 header가 있어, 해당 정보를 header에 저장
        day_data = []
        while 1:
            line = f.readline() # 첫 번째 줄 이후의 줄에는 day, hour, temperature에 대한 값이 있고 ',' 로 구분지어져 있음
            if line.strip() == '':
                break # 마지막 줄을 읽은 경우, 마지막줄은 공백이기 때문에 읽기를 종료하도록 설정
            # line = "1,200,24"
            # line.split(',') = ["1", "200", "24"]
            split_line = line.split(',')
            if len(split_line) != 3: # 1달을 넘어가는 경우, 해당 열에는 데이터의 시작 날짜가 표기되기 때문에 ','로 구분했을때 4개의 값이 존재하지 않음
                pass
            else: # ','로 구분했을때 4개의 값이 있고, 각각 순서대로
                day, hour, temperature = split_line
                day_data.append(temperature)
            if len(day_data) == 24: # 하루에 측정되는 데이터의 갯수가 24개 이므로, 24개 씩 데이터를 나눠서 저장
                day_data = np.asarray(day_data, dtype=np.float32)
                if np.min(day_data) <= -50:
                    day_data[day_data<=-50] = np.mean(day_data)
                data.append(day_data)
                day_data = []
    X = np.asarray(data, dtype=np.float32)
    X = X
    Y = np.mean(X, axis=-1) # 하루의 평균 온도를 계산
    X = X[:-1] # 마지막 날은 다음날 온도를 얻을 수 없기 때문에 제외
    Y = Y[1:] # 첫째날은 전날 정보를 얻을 수 없기 때문에 제외

    return X, Y

def train_model(X, Y):
    inputs = tf.keras.layers.Input((24, ))
    hidden1 = tf.keras.layers.Dense(32, activation='relu')(inputs)
    hidden2 = tf.keras.layers.Dense(64, activation='relu')(hidden1)
    outputs = tf.keras.layers.Dense(1)(hidden2)
    models = tf.keras.Model(inputs, outputs)

    models.compile(optimizer='adam', loss=tf.keras.losses.huber)
    models.fit(X, Y, epochs=2000)
    models.save('./data/model.h5')

if __name__ == '__main__':
    train_model(*load_data())
