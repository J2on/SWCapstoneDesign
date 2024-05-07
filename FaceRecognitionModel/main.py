import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf


# Define Intensity-aware Loss (L_IA) function
def intensity_aware_loss(y_true, y_pred):
    # Apply one-hot encoding to y_true
    num_classes = 5
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true_encoded = tf.one_hot(y_true, depth= num_classes)

    # Reshape y_true_encoded to match the shape of y_pred
    y_true_encoded = tf.reshape(y_true_encoded, (-1, num_classes))

    # Calculate PIA
    xt = y_pred * y_true_encoded  # Logits of the target class
    xmax = tf.reduce_max(y_pred * (1 - y_true_encoded), axis=1,
                         keepdims=True)  # Maximum logits excluding the target class
    numerator = tf.exp(xt)
    denominator = tf.exp(xt) + tf.exp(xmax)
    PIA = numerator / denominator

    # Calculate L_IA
    LIA = -tf.math.log(PIA)
    LIA_mean = tf.reduce_mean(LIA, axis=1)  # Compute mean LIA across classes

    # Cross-entropy loss
    cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Combine cross-entropy loss and L_IA loss
    total_loss = cross_entropy_loss + LIA_mean

    return tf.reduce_mean(total_loss)



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
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(64, activation='relu'),
    Dense(5, activation='sigmoid')  # 출력 레이어에 Sigmoid 활성화 함수 추가  처음에는 Softmax로 각 레이블 마다의 확률로 나타내보자 했지만 sigmoid로 해도 괜찮다면
])

# 모델 컴파일 <- Intensity Aware Loss 적용
model.compile(optimizer='adam', loss = intensity_aware_loss, metrics=['accuracy'])

# 모델 훈련
model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)

# 모델 평가
test_loss, test_accuracy = model.evaluate(X_scaled, y)

print("테스트 손실:", test_loss)
print("테스트 정확도:", test_accuracy)

##################################################### TEST #############################################################
file_paths_Two = ["Happy.txt", "Sad.txt", "Surprise.txt", "Normal.txt", "Disgust.txt"]

# 모든 데이터를 저장할 리스트 초기화
Normal_X = []
Y = []
# 각 파일에서 데이터를 읽어들임
lable = 0
for file_path in file_paths_Two:
    # 파일에서 데이터 읽어들임
    with open(file_path, 'r') as file:
        data_lines = file.readlines()
        for line in data_lines:

            # 각 줄에서 데이터 추출
            line_data = line.strip().split()
            # 특성과 타깃 분리
            X_line = list(map(float, line_data))
            # 리스트에 추가
            Normal_X.append(X_line)
            Y.append(float(lable))

        lable += 1

TestX_np = np.array(Normal_X)
TestY_np = np.array(Y)

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(np.array(TestX_np))

# 테스트 데이터에 대한 예측
y_pred_probs = model.predict(X_scaled)
test_loss, test_accuracy = model.evaluate(X_scaled, TestY_np)
print("테스트 손실:", test_loss)
print("테스트 정확도:", test_accuracy)


#
for y_pred_prob in y_pred_probs :
    probSum = sum(y_pred_prob)
    y_pred_prob_percent = {}
    for i in range(len(y_pred_prob)):
        y_pred_prob_percent[i] = y_pred_prob[i] * 100
    print("예측 확률:", y_pred_prob_percent)