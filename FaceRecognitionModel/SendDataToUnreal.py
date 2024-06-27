
# Socket Network
import socket
import struct

import numpy as np
from joblib import dump, load

LABEL_NUM = 7


def knn_neighbors_distribution(knn_model, X_new):
    # k개의 최근접 이웃의 인덱스 가져오기
    distances, indices = knn_model.kneighbors(X_new, n_neighbors=knn_model.n_neighbors)

    # 최근접 이웃의 레이블 가져오기
    neighbor_labels = knn_model._y[indices]

    # distributions = [0,0,0,0,0,0,0]
    # for labels in neighbor_labels:
    #     # 각 레이블의 분포 비율 계산
    #     unique, counts = np.unique(labels, return_counts=True)
    #     for i in range(len(unique)):
    #         #distributions[i] = counts[i] / knn_model.n_neighbors
    #         distributions[i] = counts[i]
    #
    # return distributions
    distributions = []
    for labels in neighbor_labels:
        # 각 레이블의 분포 비율 계산
        label_counts = np.zeros(LABEL_NUM)
        unique, counts = np.unique(labels, return_counts=True)
        label_counts[unique] = counts
        distributions.append(label_counts)

    return distributions


# Happy, Sad, Neutral, Angry, Surprise, Disgust, Fear

if __name__ == '__main__':
    # 모델 로드
    loaded_model = load('svm_model.joblib')

     # 서버 주소와 포트 설정
    HOST = 'localhost'
    PORT = 12345

    # 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    conn, addr = server_socket.accept()
    print('클라이언트가 연결됨:', addr)

    count = 0
    total_distributions = np.zeros(LABEL_NUM)

    # Unreal에서 넘겨받은 Data를 모아서 모델에 입력
    InputList = []
    received_data = np.zeros(61)
    while True :
        # Network ----------------------------------------

        # 데이터 수신
        data = conn.recv(1024).decode()
        if not data:
            break

        # 이전에 수신한 정보
        pre_received_data = received_data.copy()


        received_data = np.fromstring(data, dtype=float, sep=' ')
        received_data = received_data[0:61]

        if(np.array_equal(received_data, pre_received_data)):
            continue

        print("Data 수신")

        # 잘못된 데이터가 입력된 경우
        if(received_data.size != 61): continue

        #predictions = loaded_knn_model.predict(received_data.reshape(1, -1))
        # SendData = str(predictions)
        # # print(SendData)
        # conn.send(SendData.encode('utf-8'))
        #print(predictions)

        #distributions = knn_neighbors_distribution(loaded_knn_model, received_data.reshape(1, -1))

        # if(count < 10):
        #     #total_distributions = total_distributions + distributions
        #     # total_distributions[int(predictions[0])] += 1
        #     InputList.append(received_data)
        #     count += 1
        #     continue
        received_data = received_data.reshape(1,-1)
        predictions = loaded_model.predict_proba(received_data)
        # total_distributions[int(predictions[0])] += 1

        print("Data 송신 준비")
        for y_pred_prob in predictions:
            probSum = sum(y_pred_prob)
            y_pred_prob_percent = {}
            SendData = ''
            for i in range(len(y_pred_prob)):
                y_pred_prob_percent[i] = y_pred_prob[i] * 100
                SendData = SendData + str(y_pred_prob_percent[i]) + ' '
            print(SendData)
            conn.send(SendData.encode('utf-8'))
            print('데이터 전송 완료')


            # for dist in y_pred_prob_percent:
            #     print(dist)
            #



        InputList = []
        total_distributions = np.zeros(LABEL_NUM)
        count = 0

    conn.close()
