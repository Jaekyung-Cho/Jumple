import torch
import cv2
import socket
import time
import argparse
import numpy as np

import posenet
import jangjorim_games

import time

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=75)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.5)
parser.add_argument('--draw_skel', type=bool, default=False)
args = parser.parse_args()

HOST = '10.142.164.9'
PORT = 5036

client_socket = None

## 0~100에서 90의 이미지 품질로 설정 (default = 95)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

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


def main(game_num = '1'):

    model = posenet.load_model(args.model)
    try:
        model = model.cuda()
    except:
        model = model
    output_stride = model.output_stride

    # cv2.namedWindow("preview")
    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)

    start = time.time()
    frame_count = 0

    target = None
    part = None
    score = 0
    game_name = None
    touched = None

    # Game start delay
    time_start = time.time()
    while(time.time()-time_start < 5):
        input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

        fliped_image = cv2.flip(display_image, 1)
        fliped_image = cv2.putText(fliped_image, "Waiting for start", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        if game_num == '1':
            fliped_image = cv2.putText(fliped_image, jangjorim_games.game1.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '2':
            fliped_image = cv2.putText(fliped_image, jangjorim_games.game2.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '3':
            fliped_image = cv2.putText(fliped_image, jangjorim_games.game5.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '4':
            fliped_image = cv2.putText(fliped_image, jangjorim_games.game4.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


        cv2.imshow('posenet', fliped_image)
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    


    time_start = time.time()
    score = 0

    ball_is_here = b'F' # 볼이 여기에 있음을 표현
    ball_is_gone = b'F' # 볼이 떠났음을 표현

    # 중간에 웜홀 이미지 추가
    wormhole_img = cv2.imread("jangjorim_games/image/wormhole.png", cv2.IMREAD_UNCHANGED)
    wormhole_img = wormhole_img[:,:,:3]

    while True:
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            try:
                input_image = torch.Tensor(input_image).cuda()
            except:
                input_image = torch.Tensor(input_image)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=1,
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        if args.draw_skel:
            display_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.05, min_part_score=0.05)


        # Game Part ( modify only here !!!!!!! ) #####################################
        

        ######################################################
        # Game 1 (Catching star) #############################
        ######################################################
        if game_num == '1':
            # random target generate 
            target_image, target = jangjorim_games.game1.random_target(display_image, target, "star", "yellow")

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, touched = jangjorim_games.game1.touch_target(target_image,
                                                            keypoint_coords[0][9:11].astype(np.int32),
                                                            keypoint_coords[0][7:9].astype(np.int32),
                                                            target,
                                                            args.draw_skel)
            
            goal_score = 5
            goal_time = 10

            if touched:
                score += 1

            game_name = "Game1"

            fliped_image = cv2.flip(hand_image, 1)
            fliped_image = cv2.putText(fliped_image, "score : {}/{}".format(score, goal_score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, game_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, "{:.2f}".format(goal_time-time.time()+time_start) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if score == goal_score:
                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                            cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_image = cv2.putText(fliped_image, "Success", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return
            elif goal_time < time.time()-time_start:
                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                            cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_image = cv2.putText(fliped_image, "Failed", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return

        ######################################################  
        # Game 1 #############################################
        ######################################################



        ######################################################
        # Game 2 (falling ball touching)######################
        ######################################################

        if game_num == '2':
            # random target generate 
            target_image, target = jangjorim_games.game2.random_target(display_image, target)

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, touched = jangjorim_games.game2.touch_target(target_image,
                                                            keypoint_coords[0][9:11].astype(np.int32),
                                                            keypoint_coords[0][7:9].astype(np.int32),
                                                            target,
                                                            args.draw_skel)
            
            if touched:
                score += 1
                
            game_name = "Game2"

            fliped_image = cv2.flip(hand_image, 1)
            # Score function
            fliped_image = cv2.putText(fliped_image, "score : "+str(score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, game_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            if score == jangjorim_games.game2.num_of_target:
                success_time = "Time : " + str(round(time.time() - time_start,2))
                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                            cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_image = cv2.putText(fliped_image, "Success", (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    fliped_image = cv2.putText(fliped_image, success_time , (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return

        ######################################################  
        # Game 2 #############################################
        ######################################################



        ######################################################
        # Game 3 (Save Soccerball) ###########################
        ######################################################

        if game_num == '3':
            # Range of hit
            height = display_image.shape[0]
            hit_range = [
                ("Perfect", int(height/2-height/20), int(height/2+height/20)),
                ("Great", int(height/2-height/8), int(height/2+height/8)),
                ("Bad", int(height/2-height/2), int(height/2+height/2))
            ]

            # random target generate 
            target_image, target = jangjorim_games.game3.random_target(display_image, target, "soccerball", "original")

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, touched, scored= jangjorim_games.game3.touch_target(target_image,
                                                            keypoint_coords[0][9:11].astype(np.int32),
                                                            keypoint_coords[0][7:9].astype(np.int32),
                                                            target,
                                                            hit_range,
                                                            touched,
                                                            args.draw_skel)
            if touched == None:
                score = 0
            else:
                score += scored

            game_name = "Volley ball challenge"

            fliped_image = cv2.flip(hand_image, 1)

            #
            fliped_image = cv2.line(fliped_image, (30,int(height/2)),(80,int(height/2)),(0,0,255),3)
            # Score function
            fliped_image = cv2.putText(fliped_image, touched, (300, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, game_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, "score : "+str(score), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)


        ######################################################  
        # Game 3 #############################################
        ######################################################


        ######################################################
        # Game 4 (English touching) ##########################
        ######################################################
        if game_num == '4':
            # random target generate 
            target_image, target, part = jangjorim_games.game4.random_target(display_image, target, part, "circle", "yellow")

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, part, touched = jangjorim_games.game4.touch_target(target_image,
                                                            keypoint_coords[0].astype(np.int32),
                                                            target,
                                                            part,
                                                            args.draw_skel)
            
            goal_score = 10
            goal_time = 100

            if touched:
                score += 1

            game_name = "Game4"

            fliped_image = cv2.flip(hand_image, 1)
            if target is not None:
                fliped_image = cv2.putText(fliped_image, part, (fliped_image.shape[1]-target[0]-30,target[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
            fliped_image = cv2.putText(fliped_image, "score : {}/{}".format(score, goal_score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, game_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            fliped_image = cv2.putText(fliped_image, "{:.2f}".format(goal_time-time.time()+time_start) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if score == goal_score:
                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                            cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_image = cv2.putText(fliped_image, "Success", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return
            elif goal_time < time.time()-time_start:
                while True:
                    input_image, display_image, output_scale = posenet.read_cap(
                            cap, scale_factor=args.scale_factor, output_stride=output_stride)

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_image = cv2.putText(fliped_image, "Failed", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

                    cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return

        ######################################################  
        # Game 4 #############################################
        ######################################################


        ######################################################
        # Game 5 (volley ball together) ###########################
        ######################################################

        if game_num == '5':

            
            # Range of hit
            height = display_image.shape[0]
            # hit_range = [
            #     ("Perfect", int(height/2-height/20), int(height/2+height/20)),
            #     ("Great", int(height/2-height/8), int(height/2+height/8)),
            #     ("Bad", int(height/2-height/2), int(height/2+height/2))
            # ]

            ball_is_gone = b'F'
            if ball_is_here == b'T':

              # random target generate 
              display_image, target, gone = jangjorim_games.game5.random_target(
                display_image, target, time.time()-time_start,"soccerball", "original")
              time_start = time.time()

              if gone is None:
                score -= 1
              elif gone:
                ball_is_gone = b'T'
              else:
                ball_is_gone = b'F'
              

              # index 9 = left wrist, index 10 = right wrist
              # check if hand close to target    
              display_image, target, scored= jangjorim_games.game5.touch_target(display_image,
                                                              keypoint_coords[0][9:11].astype(np.int32),
                                                              keypoint_coords[0][7:9].astype(np.int32),
                                                              target,
                                                              args.draw_skel)

              # score 계산 ( 뒷빡 없음 )                                            
              score += scored
              if score < 0 :
                score = 0

            game_name = "Volley ball challenge"

            fliped_image = cv2.flip(display_image, 1)

            
                    # image server로 보낼 때 downsampling
            down_img = cv2.resize(fliped_image, dsize=(0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
            
            
            # Score function (점수 표시 하기)
            fliped_image = cv2.putText(fliped_image, "score : "+str(score), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
            fliped_image = cv2.resize(fliped_image, dsize=(int(args.cam_width*2/3), int(args.cam_height*2/3)))

            # cv2. imencode(ext, img [, params])
            # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
            result, frame = cv2.imencode('.jpg', fliped_image, encode_param)
            # frame을 String 형태로 변환
            data = np.array(frame)
            stringData = data.tostring()

            # 공이 있는지 없는지 판단하는 거 뒤에다가 붙여줌
            stringData += ball_is_here + ball_is_gone
        
            #서버에 데이터 전송
            #(str(len(stringData))).encode().ljust(16)
            client_socket.sendall((str(len(stringData))).encode().ljust(16) + stringData)

            print("get_data")
            # 서버에서 통합된 영상 받아서 화면에 내보내기
            length = recvall(client_socket, 16)
            stringData = recvall(client_socket, int(length))

            # 공이 있는지 없는지 저장해주는 것
            ball_is_here = stringData[-1:]
            stringData = stringData[:-1]

            data = np.fromstring(stringData, dtype = 'uint8')
            #data를 디코딩한다.
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            # 높이를 480으로 고정하기 위한 작업
            h,w,_ = fliped_image.shape
            # image upsampling
            mid_img = cv2.resize(wormhole_img, dsize = (wormhole_img.shape[1],h))
            frame = cv2.resize(frame, dsize=(w,h), interpolation=cv2.INTER_CUBIC)
            final = cv2.hconcat([fliped_image, mid_img])
            fliped_image = cv2.hconcat([final, frame])

        ######################################################  
        # Game 5 #############################################
        ######################################################



        ################################################################################

        
        # cv2.namedWindow('posenet', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('posenet', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('posenet', fliped_image)
        cv2.imshow('posenet', fliped_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('0'):
            return

    print('Average FPS: ', frame_count / (time.time() - start))


if __name__ == "__main__":
    try:
        while True:

            blank_img = np.zeros((480,720,3),np.uint8)

            blank_img = cv2.putText(blank_img, "Press button for start game",(30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            blank_img = cv2.putText(blank_img, "1 : Catching Stars",(30,105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            blank_img = cv2.putText(blank_img, "2 : Pop Ballon",(30,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            blank_img = cv2.putText(blank_img, "3 : Volley Ball Alone",(30,195), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            blank_img = cv2.putText(blank_img, "4 : English ",(30,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            blank_img = cv2.putText(blank_img, "5 : volley together",(30,285), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            blank_img = cv2.putText(blank_img, "0 : Quit Game",(30,330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            
            time.sleep(0.1)
            cv2.imshow('posenet', blank_img)

            game_type = None
            while True:
                key = cv2.waitKey()
                if key == ord('1'):
                    game_type = '1'
                    break
                if key == ord('2'):
                    game_type = '2'
                    break
                if key == ord('3'):
                    game_type = '3'
                    break
                if key == ord('4'):
                    game_type = '4'
                    break
                if key == ord('5'):
                    game_type = '5'
                    client_socket = socket.socket(socket.AF_INET,socket.SOCK_STREAM) 
                    client_socket.connect((HOST, PORT)) 
                    break
                if key == ord('q'):
                    raise getOutOfLoop

            main(game_type)
    
    except getOutOfLoop:
        pass