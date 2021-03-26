import numpy as np
import socket 
from _thread import *
import cv2
import threading

# TCP port 
HOST = input("Insert Host IP address : ")
PORT = 5036

# black image for default
# image size
size = (160,120)


#socket에서 수신한 버퍼를 반환하는 함수
def recvall(sock, count):
    # 바이트 문자열
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf



client_dict = {}
ball_dict = {}
score_dict = {}
client_addrs = []
clients_lock = threading.Lock()

# 쓰레드에서 실행되는 코드입니다. 
# 접속한 클라이언트마다 새로운 쓰레드가 생성되어 통신을 하게 됩니다. 
def threaded(client_socket, addr): 
  global concated_img
  global size
  # 목록에 추가
  
  print('Connected by :', addr[0], ':', addr[1]) 
  if len(client_addrs)==0:
    ball_dict[addr[1]] = b'T'
  else:
    ball_dict[addr[1]] = b'F'
  client_addrs.append(addr[1])
  score_dict[addr[1]] = b'F'

  # 클라이언트가 접속을 끊을 때 까지 반복합니다. 
  while True: 
    # 두명이 접속했을 때에만 동작합니다.
    if len(client_addrs) ==2:
      try:
        length = recvall(client_socket, 16)
        if length is not None:
          stringData = recvall(client_socket, int(length))

          # 볼이 있는지 없는지 판단
          ball_is_here = ball_dict[addr[1]]
          ball_is_gone = stringData[-1:]
          scored = stringData[-3:-2]
          stringData = stringData[:-3]
          
          # 볼 위치 변경
          if ball_is_gone == b'T':
            if len(client_addrs) ==2:
              for key in client_addrs:
                if ball_dict[key] == b'T':
                  ball_dict[key] = b'F'
                else:
                  ball_dict[key] = b'T'
          if scored == b'T':
            for key in client_addrs:
              if key is not addr[1]:
                score_dict[key] = b'T'


          data = np.fromstring(stringData, dtype = 'uint8')
          #data를 디코딩한다.
          frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
          # client 데이터들 dictionary에 모으기
          client_dict[addr[1]] = frame


          img = np.zeros((size[1],size[0]),np.uint8)
          for key in client_addrs:
            if key != addr[1]:
              if key in client_dict:
                img = cv2.resize(client_dict[key], dsize=size)
                img = cv2.flip(img, 1)

          # cv2. imencode(ext, img [, params])
          # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
          result, frame = cv2.imencode('.jpg', img, encode_param)
          # frame을 String 형태로 변환
          stringData = frame.tostring()
          
          # ball이 어디있는지 표현
          scored = score_dict[addr[1]]
          stringData += scored + ball_dict[addr[1]]
          if scored == b'T':
            score_dict[addr[1]] = b'F'
      
          #서버에 데이터 전송
          #(str(len(stringData))).encode().ljust(16)
          client_socket.sendall((str(len(stringData))).encode().ljust(16) + stringData)
          print("data sending")

        # client_socket.send(data) 
      except ConnectionResetError as e:
        print('Disconnected by ' + addr[0],':',addr[1])
        # socket 통신 종료
        client_socket.close() 
        # client dict 에서 삭제
        del client_dict[addr[1]]
        del ball_dict[addr[1]]
        client_addrs.remove(addr[1])
        break




server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) 
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT)) 
# 2개까지만 client 받는다
server_socket.listen(2) 

print('server start')

## 0~100에서 90의 이미지 품질로 설정 (default = 95)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

# 클라이언트가 접속하면 accept 함수에서 새로운 소켓을 리턴합니다.

# 새로운 쓰레드에서 해당 소켓을 사용하여 통신을 하게 됩니다. 
while True: 
  print('wait')

  client_socket, addr = server_socket.accept() 
  start_new_thread(threaded, (client_socket, addr)) 

server_socket.close() 