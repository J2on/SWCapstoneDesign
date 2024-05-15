
# Socket Network
import socket
import struct

import numpy as np
from joblib import dump, load

if __name__ == '__main__':
    # 모델 로드
    loaded_knn_model = load('knn_model.joblib')

     # 서버 주소와 포트 설정
    HOST = 'localhost'
    PORT = 12345

    # 소켓 생성
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    conn, addr = server_socket.accept()
    print('클라이언트가 연결됨:', addr)

    while True :
        # Network ----------------------------------------

        # 데이터 수신
        data = conn.recv(1024).decode()
        if not data:
            break
        received_data = np.fromstring(data, dtype=float, sep=' ')
        received_data = received_data[0:61]

        print("Data 수신")
        # 잘못된 데이터가 입력된 경우
        if(received_data.size != 61): continue

        print("Data 송신 준비")

        predictions = loaded_knn_model.predict(received_data.reshape(1, -1))
        SendData = str(predictions)
        print(SendData)
        conn.send(SendData.encode('utf-8'))

        print('데이터 전송 완료')
    conn.close()
