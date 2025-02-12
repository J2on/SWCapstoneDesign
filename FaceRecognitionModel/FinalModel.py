import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load

def knn_class_probabilities(knn_model, X_new):
    # KNN 모델을 사용하여 클래스 레이블을 예측합니다.
    predicted_labels = knn_model.predict(X_new)

    # 예측된 클래스 레이블에 대한 확률을 계산합니다.
    class_probabilities = np.zeros((X_new.shape[0], len(knn_model.classes_)))
    for i, label in enumerate(knn_model.classes_):
        class_probabilities[:, i] = (predicted_labels == label).mean(axis=0)

    return class_probabilities


def knn_neighbors_distribution(knn_model, X_new):
    # k개의 최근접 이웃의 인덱스 가져오기
    distances, indices = knn_model.kneighbors(X_new, n_neighbors=knn_model.n_neighbors)

    # 최근접 이웃의 레이블 가져오기
    neighbor_labels = knn_model._y[indices]

    distributions = [0,0,0,0,0,0,0]
    for labels in neighbor_labels:
        # 각 레이블의 분포 비율 계산
        unique, counts = np.unique(labels, return_counts=True)
        for i in range(len(unique)):
            distributions[i] = counts[i] / knn_model.n_neighbors

    return distributions

# 모델 로드
loaded_knn_model = load('knn_model.joblib')

file_paths_Two = ["Happy.txt", "Sad.txt"]
#file_paths_Two = ["Happy.txt", "Sad.txt", "Surprise.txt", "Normal.txt", "Disgust.txt"]

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

            Normal_X = []
            Y = []
            # 각 줄에서 데이터 추출
            line_data = line.strip().split()
            # 특성과 타깃 분리
            X_line = list(map(float, line_data))
            # 리스트에 추가
            Normal_X.append(X_line)
            Y.append(float(lable))
            class_probabilities = knn_class_probabilities(loaded_knn_model, np.array(Normal_X))
            distances, indices = loaded_knn_model.kneighbors(Normal_X)
            distributions = knn_neighbors_distribution(loaded_knn_model, Normal_X)
            print(distributions)
            #print("Class Probabilities:", class_probabilities, " Label : ", lable)

        lable += 1
#
# 예측
accuracy = loaded_knn_model.score(Normal_X, Y)
print("Test set accuracy: {:.2f}".format(accuracy))

predictions = loaded_knn_model.predict(Normal_X)
print(predictions)




# # 모델 사용 예시
# predictions = loaded_knn_model.predict(Normal_X)
# print(predictions)
