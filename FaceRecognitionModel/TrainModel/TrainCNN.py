import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Dropout, MaxPooling1D
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import matplotlib.pyplot as plt

@tf.keras.utils.register_keras_serializable()
def intensity_aware_loss(y_true, y_pred):
    num_classes = 7  # 7 classes in total

    # Ensure y_true and y_pred have the same batch size
    batch_size = tf.shape(y_pred)[0]
    y_true = tf.slice(y_true, [0, 0], [batch_size, -1])

    # Calculate P_IA
    x_t = y_pred * y_true  # Logits of the target class
    x_max = tf.reduce_max(y_pred * (1 - y_true), axis=1, keepdims=True)  # Maximum logits excluding the target class
    numerator = tf.exp(x_t)
    denominator = tf.exp(x_t) + tf.exp(x_max)
    P_IA = numerator / denominator

    # Calculate IntensityAwareLoss
    IntensityAwareLoss = -tf.math.log(P_IA)
    IntensityAwareLoss_mean = tf.reduce_mean(IntensityAwareLoss, axis=1)  # Compute mean IntensityAwareLoss across classes

    # Cross-entropy loss
    cross_entropy_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Combine cross-entropy loss and IntensityAwareLoss loss
    hyperParameter = 0.5
    total_loss = cross_entropy_loss + hyperParameter * IntensityAwareLoss_mean

    return tf.reduce_mean(total_loss)


filePaths = []

filePathHappy = [f"Data/Happy/Happy{i}.txt" for i in range(175)]
filePathSad = [f"Data/Sad/Sad{i}.txt" for i in range(169)]
filePathNeutral = [f"Data/Neutral/Neutral{i}.txt" for i in range(199)]
filePathAngry = [f"Data/Angry/Angry{i}.txt" for i in range(146)]
filePathSurprise = [f"Data/Surprise/Surprise{i}.txt" for i in range(89)]
filePathDisgust = [f"Data/Disgust/Disgust{i}.txt" for i in range(8)]
filePathFear = [f"Data/Fear/Fear{i}.txt" for i in range(50)]

filePaths.extend([filePathHappy, filePathSad, filePathNeutral, filePathAngry, filePathSurprise, filePathDisgust, filePathFear])

# 모든 데이터를 저장할 리스트 초기화
all_X = []
all_y = []

lable = 0
for path_Emotion in filePaths:
    for filePath in path_Emotion:
        # 파일에서 데이터 읽어들임
        with open(filePath, 'r') as file:
            data_lines = file.readlines()
            for line in data_lines:
                # 각 줄에서 데이터 추출
                line_data = line.strip().split()
                # 특성과 타깃 분리
                X_line = list(map(float, line_data))
                y_line = float(lable)
                # 리스트에 추가
                all_X.append(X_line[0:61])
                all_y.append(y_line)
    lable += 1

# 리스트를 numpy 배열로 변환
X = np.array(all_X)
y = np.array(all_y)

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1D CNN을 위해 입력 데이터의 형태를 (samples, timesteps, features)로 변경
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# 레이블을 원-핫 인코딩
num_classes = 7  # 클래스 수 (Happy, Sad, Neutral, Angry, Surprise, Disgust, Fear)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# CNN 모델 생성
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam', loss=intensity_aware_loss, metrics=['accuracy'])

# 모델 학습
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 모델 평가
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# 학습 곡선 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over Epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over Epochs')

plt.show()

# 테스트 데이터 예측
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# 평가 보고서 출력
report = classification_report(y_test_labels, y_pred)
print("Classification Report:\n", report)

# 모델 저장 (SavedModel 형식)
model.save('saved_model.keras')

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

############################################################################
TestX_np = np.array(Normal_X)
TestY_np = np.array(Y)

# 1D CNN을 위해 입력 데이터의 형태를 (samples, timesteps, features)로 변경
TestX_np = np.expand_dims(TestX_np, axis=2)

# 테스트 데이터 예측
y_pred_probs = model.predict(TestX_np)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(TestY_np, y_pred)
report = classification_report(TestY_np, y_pred)

print("테스트 정확도:", accuracy)
print("Classification Report:\n", report)

cnt = 1
for y_pred_prob in y_pred_probs:
    probSum = sum(y_pred_prob)
    y_pred_prob_percent = {i: y_pred_prob[i] * 100 for i in range(len(y_pred_prob))}
    print(f"{cnt} 번째 예측 확률:", y_pred_prob_percent)
    cnt += 1
##############################################################################
