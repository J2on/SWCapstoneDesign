import random

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load


# Define Intensity-aware Loss (IntensityAwareLoss) function
def intensity_aware_loss(y_true, y_pred):
    num_classes = 7

    # Apply one-hot encoding to y_true
    y_true = tf.cast(y_true, dtype=tf.int32)
    y_true_encoded = tf.one_hot(y_true, depth= num_classes)

    # Reshape y_true_encoded to match the shape of y_pred
    y_true_encoded = tf.reshape(y_true_encoded, (-1, num_classes))

    # Calculate P_IA
    x_t = y_pred * y_true_encoded  # Logits of the target class
    x_max = tf.reduce_max(y_pred * (1 - y_true_encoded), axis=1,
                         keepdims=True)  # Maximum logits excluding the target class
    numerator = tf.exp(x_t)
    denominator = tf.exp(x_t) + tf.exp(x_max)
    P_IA = numerator / denominator

    # Calculate IntensityAwareLoss
    IntensityAwareLoss = -tf.math.log(P_IA)
    IntensityAwareLoss_mean = tf.reduce_mean(IntensityAwareLoss, axis=1)  # Compute mean IntensityAwareLoss across classes 더해줌

    # Cross-entropy loss
    cross_entropy_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

    # Combine cross-entropy loss and IntensityAwareLoss loss // Cross Entropy +
    hyperParmeter = 0.5
    total_loss = cross_entropy_loss + hyperParmeter * IntensityAwareLoss_mean

    return tf.reduce_mean(total_loss)




filePaths = []



filePathHappy = []
for i in range (0,175):
    filePathHappy.append(f"Data/Happy/Happy{i}.txt");
filePathSad = []
for i in range (0,169):
    filePathSad.append(f"Data/Sad/Sad{i}.txt");
filePathNeutral = []
for i in range(0, 199):
    filePathNeutral.append(f"Data/Neutral/Neutral{i}.txt");
filePathAngry = []
for i in range(0, 146):
    filePathAngry.append(f"Data/Angry/Angry{i}.txt");
filePathSurprise = []
for i in range (0,89):
    filePathSurprise.append(f"Data/Surprise/Surprise{i}.txt");
filePathDisgust = []
for i in range (0,8):
    filePathDisgust.append(f"Data/Disgust/Disgust{i}.txt");
filePathFear = []
for i in range (0,50):
    filePathFear.append(f"Data/Fear/Fear{i}.txt");




filePaths.append(filePathHappy)
filePaths.append(filePathSad)
filePaths.append(filePathNeutral)
filePaths.append(filePathAngry)
filePaths.append(filePathSurprise)
filePaths.append(filePathDisgust)
filePaths.append(filePathFear)

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

svm_model = SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

# 모델 평가
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)

# 모델 저장
dump(svm_model, 'svm_model.joblib')

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

# 테스트 데이터에 대한 예측
y_pred_probs = svm_model.predict_proba(TestX_np)
y_pred = np.argmax(y_pred_probs, axis=1)

accuracy = accuracy_score(TestY_np, y_pred)
report = classification_report(TestY_np, y_pred) # 평가지표

print("테스트 정확도:", accuracy)
print("Classification Report:\n", report)

cnt = 1;
for y_pred_prob in y_pred_probs:
    probSum = sum(y_pred_prob)
    y_pred_prob_percent = {}
    for i in range(len(y_pred_prob)):
        y_pred_prob_percent[i] = y_pred_prob[i] * 100
    print(str(cnt) + " 번째 예측 확률:", y_pred_prob_percent)
    cnt += 1
##############################################################################
