import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load




# 데이터 파일 경로


filePaths = []

filePathHappy = []
for i in range (0,50):
    filePathHappy.append(f"Data/Happy/Happy{i}.txt");
filePathSad = []
for i in range (0,50):
    filePathSad.append(f"Data/Sad/Sad{i}.txt");
filePathNeutral = []
for i in range(0, 50):
    filePathNeutral.append(f"Data/Neutral/Neutral{i}.txt");
filePathAngry = []
for i in range(0, 50):
    filePathAngry.append(f"Data/Angry/Angry{i}.txt");
filePathSurprise = []
for i in range (0,50):
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


#file_paths = ["Happy.txt", "Sad.txt", "Surprise.txt", "Normal.txt", "Disgust.txt"]
# 모든 데이터를 저장할 리스트 초기화
all_X = []
all_y = []

lable = 0

# 각 파일에서 데이터를 읽어들임
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
Y = np.array(all_y)

# 데이터 분할 (훈련 세트와 테스트 세트)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)  # K값은 5로 설정합니다. 다른 값을 선택할 수도 있습니다.

# 모델 훈련
knn_model.fit(X_train, y_train)

# 모델 평가
accuracy = knn_model.score(X_test, y_test)
print("Test set accuracy: {:.2f}".format(accuracy))

# 모델 저장
dump(knn_model, 'Small_knn_model.joblib')


##################################################### TEST #############################################################
#file_paths_Two = ["Happy.txt", "Sad.txt", "Surprise.txt", "Normal.txt", "Disgust.txt"]


