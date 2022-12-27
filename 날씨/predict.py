import tensorflow as tf
import numpy as np

TABLE_DATA = """시:분	강수	강수15	강수60	강수3H	강수6H	강수12H	일강수	기온	풍향1	풍속1(m/s)	풍향10	풍속10(m/s)
23:00	○	0	0	0	0	0	0	-10.3	334.8	NNW	1.2	328.5	NNW	1.6
22:00	○	0	0	0	0	0	0	-10.1	328.4	NNW	2.2	320.1	NW	2.5
21:00	○	0	0	0	0	0	0	-9.9	312.6	NW	3.7	310.3	NW	2.2
20:00	○	0	0	0	0	0	0	-9.6	321.3	NW	2.2	311.6	NW	2.7
19:00	○	0	0	0	0	0	0	-9.3	319.7	NW	4.2	309.2	NW	3.1
18:00	○	0	0	0	0	0	0	-8.9	315.0	NW	2.6	314.3	NW	3.6
17:00	○	0	0	0	0	0	0	-8.3	327.7	NNW	2.7	314.7	NW	3.7
16:00	○	0	0	0	0	0	0	-8.0	321.5	NW	1.7	317.4	NW	2.5
15:00	○	0	0	0	0	0	0	-8.7	331.4	NNW	3.4	313.4	NW	3.8
14:00	○	0	0	0	0	0	0	-8.7	314.2	NW	3.5	318.8	NW	3.8
13:00	○	0	0	0	0	0	0	-9.1	316.5	NW	2.4	312.0	NW	2.4
12:00	○	0	0	0	0	0	0	-9.8	298.5	WNW	3.9	321.7	NW	2.6
11:00	○	0	0	0	0	0	0	-10.2	125.1	SE	1.0	354.0	N	0.7
10:00	○	0	0	0	0	0	0	-12.5	116.2	ESE	1.2	143.5	SE	1.5
09:00	○	0	0	0	0	0	0	-14.5	156.6	SSE	1.1	136.8	SE	1.0
08:00	○	0	0	0	0	0	0	-16.1	182.8	S	1.5	169.6	S	1.1
07:00	○	0	0	0	0	0	0	-15.2	163.2	SSE	1.2	194.5	SSW	0.8
06:00	○	0	0	0	0	0	0	-14.4	289.4	WNW	1.5	324.7	NW	1.6
05:00	○	0	0	0	0	0	0	-17.0	335.7	NNW	0.8	141.4	SE	0.9
04:00	○	0	0	0	0	0	0	-17.0	130.4	SE	0.8	160.0	SSE	1.0
03:00	○	0	0	0	0	0	0	-14.3	322.7	NW	0.5	315.8	NW	1.0
02:00	○	0	0	0	0	0	0	-14.4	331.5	NNW	0.4	331.5	NNW	0.8
01:00	○	0	0	0	0	0	0	-14.5	33.5	NNE	1.6	66.8	ENE	0.8
00:00	○	0	0	0	0	0	0	-14.4	302.6	WNW	1.6	319.3	NW	1.9"""
MODEL_PATH = './data/model.h5'


def extract_temperature_from_table():
    data = TABLE_DATA.split('\n')
    data = data[1:]
    data = [float(x.split('\t')[8]) for x in data[::-1]]
    return np.asarray(data)


def forecast(temperature):
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    predict = model.predict(temperature[np.newaxis])[0][0]
    if predict > 30:
        print("날씨가 많이 더울 예정입니다. 야외활동에 주의하세요")
    elif predict > 25:
        print("날씨가 더울 예정입니다. 옷을 가볍게 입고 나가세요")
    elif predict > 15:
        print("야외활동 하기 좋은 날씨입니다. 책상을 벗어나는 하루를 만들어 보는건 어떨까요")
    elif predict > 5:
        print("선선한 날씨입니다. 가벼운 외투 하나더 챙겨 입는것을 추천드립니다")
    elif predict > -5:
        print("날씨가 추울 예정입니다. 옷을 두껍게 입고 나가세요")
    elif predict > -15:
        print("날씨가 많이 추울 예정입니다. 야외활동에 주의하세요")


if __name__ == '__main__':
    temperature = extract_temperature_from_table()
    forecast(temperature)

