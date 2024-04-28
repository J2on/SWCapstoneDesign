import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 데이터 파일 경로
file_paths = ["Happy.txt", "Sad.txt", "Surprise.txt", "Normal.txt", "Disgust.txt"]

# 모든 데이터를 저장할 리스트 초기화
all_X = []
all_y = []

lable = 0
# 각 파일에서 데이터를 읽어들임
for file_path in file_paths:
    # 파일에서 데이터 읽어들임
    with open(file_path, 'r') as file:
        data_lines = file.readlines()
        for line in data_lines:
            # 각 줄에서 데이터 추출
            line_data = line.strip().split()
            # 특성과 타깃 분리
            X_line = list(map(float, line_data))
            y_line = float(lable)
            # 리스트에 추가
            all_X.append(X_line)
            all_y.append(y_line)
    lable += 1

# 리스트를 numpy 배열로 변환
X = np.array(all_X)
y = np.array(all_y)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 모델 생성
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(5, activation='softmax')  # 출력 레이어에 Softmax 활성화 함수 추가
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print("테스트 손실:", test_loss)
print("테스트 정확도:", test_accuracy)

# # 테스트 데이터에 대한 예측
# y_pred_probs = model.predict(X_test)
#
# for y_pred_prob in y_pred_probs :
#     probSum = sum(y_pred_prob)
#     y_pred_prob_percent = {}
#     for i in range(len(y_pred_prob)):
#         y_pred_prob_percent[i] = (y_pred_prob[i]/probSum) * 100
#     print("예측 확률:", y_pred_prob_percent)

print('123123123123123123=-----------------------------------------------------------------------------------')



file_paths = ["Normal.txt"]

# 모든 데이터를 저장할 리스트 초기화
Normal_X = []

lable = 0
# 각 파일에서 데이터를 읽어들임
for file_path in file_paths:
    # 파일에서 데이터 읽어들임
    with open(file_path, 'r') as file:
        data_lines = file.readlines()
        for line in data_lines:
            # 각 줄에서 데이터 추출
            line_data = line.strip().split()
            # 특성과 타깃 분리
            X_line = list(map(float, line_data))
            y_line = float(lable)
            # 리스트에 추가
            Normal_X.append(X_line)
    lable += 1

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(Normal_X))

# 테스트 데이터에 대한 예측
y_pred_probs = model.predict(X_scaled)

for y_pred_prob in y_pred_probs :
    probSum = sum(y_pred_prob)
    y_pred_prob_percent = {}
    for i in range(len(y_pred_prob)):
        y_pred_prob_percent[i] = (y_pred_prob[i]/probSum) * 100
    print("예측 확률:", y_pred_prob_percent)










# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# from sklearn.model_selection import train_test_split
#
# # 가상의 데이터 생성
# X = np.random.rand(1000000, 30)  # 1000개의 샘플과 30개의 특성 생성
# y = np.random.randint(5, size=1000000)  # 5가지 클래스 레이블 생성
#
# # 데이터를 훈련 세트와 테스트 세트로 나눔
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # 신경망 모델 생성
# model = Sequential([
#     Dense(64, activation='relu', input_shape=(30,)),
#     Dense(64, activation='relu'),
#     Dense(5, activation='softmax')
# ])
#
# # 모델 컴파일
# model.compile(optimizer='adam',
#               loss=SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
#
# # 모델 훈련
# model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
#
# # 테스트 세트로 모델 평가
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print('테스트 손실:', test_loss)
# print('테스트 정확도:', test_accuracy)