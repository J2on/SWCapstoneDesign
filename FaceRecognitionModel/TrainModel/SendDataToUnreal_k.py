# Socket Network
import socket
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

LABEL_NUM = 7

# Define Intensity-aware Loss (IntensityAwareLoss) function
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

if __name__ == '__main__':
    # 모델 로드
    loaded_model = load_model('saved_model.keras')  # 여기에서 joblib의 load 대신 keras의 load_model을 사용합니다.

    # 서버 주소와 포트 설정
    HOST = 'localhost'
    PORT = 12345

    # 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    conn, addr = server_socket.accept()
    print('클라이언트가 연결됨:', addr)

    received_data = np.zeros(61)
    while True:
        # 데이터 수신
        data = conn.recv(1024).decode()
        if not data:
            break

        # 이전에 수신한 정보
        pre_received_data = received_data.copy()

        received_data = np.fromstring(data, dtype=float, sep=' ')
        received_data = received_data[0:61]

        if np.array_equal(received_data, pre_received_data):
            continue

        print("Data 수신")

        # 잘못된 데이터가 입력된 경우
        if received_data.size != 61:
            continue

        # 데이터 형태를 (samples, timesteps, features)로 변경
        received_data = received_data.reshape(1, -1, 1)
        predictions = loaded_model.predict(received_data)

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

    conn.close()
