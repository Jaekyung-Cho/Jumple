import torch
import cv2
import socket
import time
import argparse
import numpy as np
from PIL import Image, ImageFont, ImageDraw

img_rate = 0.8

import posenet
import jangjorim_games

import time
from playsound import playsound
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

font_dir = resource_path("jangjorim_games/font/VT323-Regular.ttf")

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=75)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=640)
parser.add_argument('--cam_height', type=int, default=480)
parser.add_argument('--scale_factor', type=float, default=0.5)
parser.add_argument('--draw_skel', type=bool, default=False)
parser.add_argument('--host_ip', type=str, default="127.0.0.1")
args = parser.parse_args()


skel_draw = None

HOST = None
PORT = 5036

client_socket = None

## 0~100에서 90의 이미지 품질로 설정 (default = 95)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

cap = cv2.VideoCapture(args.cam_id)
cap.set(3, args.cam_width)
cap.set(4, args.cam_height)

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

def image_overlapping(screen, img, pos):
    x_size, y_size, _ = img.shape
    x1, x2 = pos[0], pos[0]+x_size
    y1, y2 = pos[1], pos[1]+y_size
    if y2 >= screen.shape[1]:
        y2 = screen.shape[1]-1
        y1 = y2 - y_size
    if y1 < 0:
        y1 = 0
        y2 = y1 + y_size
    if x2 >= screen.shape[0]:
        x2 = screen.shape[0]-1
        x1 = x2 - x_size
    if x1 < 0:
        x1 = 0
        x2 = x1 + x_size

    alpha_img = img[:,:,3]/255.0
    alpha_screen = 1- alpha_img

    for c in range(3):
        screen[x1:x2, y1:y2, c] = alpha_img * img[:,:,c] + alpha_screen * screen[x1:x2, y1:y2, c]

    return screen

def text_overlapping(screen, text, pos, size, color, center=True):
    pill_image = Image.fromarray(cv2.cvtColor(screen, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pill_image)
    if center:
        w,h = draw.textsize(text, font=ImageFont.truetype(font_dir, size))
        draw.text((pos[0]-int(w/2), pos[1]-int(h/2)), text, font=ImageFont.truetype(font_dir, size), fill=color)
    else:
        draw.text(pos, text, font=ImageFont.truetype(font_dir, size), fill=(255, 255, 255))
    output = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)

    return output


def main_loop(game_num = '1'):

    model = posenet.load_model(args.model)
    try:
        model = model.cuda()
    except:
        model = model
    output_stride = model.output_stride

    # cv2.namedWindow("preview")
    

    start = time.time()
    frame_count = 0

    target = None
    part = None
    score = 0
    game_name = None
    touched = None
    touch_img = None

    # for game 5
    my_score = 0
    another_score = 0


    sound_mario = resource_path("jangjorim_games/sound/mario.mp3")
    sound_mario = sound_mario.replace(" ", "%20")
    # Game start delay
    time_start = time.time()
    while(time.time()-time_start < 5):
        while(True):
            try:
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride)
                break
            except:
                cap = cv2.VideoCapture(args.cam_id)
                cap.set(3, args.cam_width)
                cap.set(4, args.cam_height)
                time.sleep(0.1)
                pass
            
        
        fliped_image = cv2.flip(display_image, 1)
        rate_w = 640.0/fliped_image.shape[1]
        rate_h = 480.0/fliped_image.shape[0]
        rate = rate_w if rate_w > rate_h else rate_h
        fliped_iamge = cv2.resize(fliped_image, dsize = (0,0), fx = rate, fy = rate)
        fliped_image = fliped_image[0:480 ,0:640]
        fliped_image = cv2.resize(fliped_image, dsize=(640,480))
        
        # frame 142 addition
        frame_142 = cv2.imread(resource_path("jangjorim_games/image/Frame 142.png"), cv2.IMREAD_UNCHANGED)
        frame_142 = cv2.resize(frame_142, dsize=(int(800*img_rate),int(40*img_rate)))
        fliped_image = image_overlapping(fliped_image, frame_142, (int(530*img_rate),0))

        # fliped_image = cv2.putText(fliped_image, "Waiting for start", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        if game_num == '1':
            frame_w = cv2.imread(resource_path("jangjorim_games/image/Frame 168.png"), cv2.IMREAD_UNCHANGED)
            frame_w = cv2.resize(frame_w, dsize=(int(600*img_rate),int(171*img_rate)))
            # fliped_image = cv2.putText(fliped_image, jangjorim_games.game1.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '2':
            frame_w = cv2.imread(resource_path("jangjorim_games/image/Frame 169.png"), cv2.IMREAD_UNCHANGED)
            frame_w = cv2.resize(frame_w, dsize=(int(600*img_rate),int(171*img_rate)))
            # fliped_image = cv2.putText(fliped_image, jangjorim_games.game2.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '3':
            frame_w = cv2.imread(resource_path("jangjorim_games/image/Frame 170.png"), cv2.IMREAD_UNCHANGED)
            frame_w = cv2.resize(frame_w, dsize=(int(600*img_rate),int(171*img_rate)))
            # fliped_image = cv2.putText(fliped_image, jangjorim_games.game5.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '4':
            frame_w = cv2.imread(resource_path("jangjorim_games/image/Frame 171.png"), cv2.IMREAD_UNCHANGED)
            frame_w = cv2.resize(frame_w, dsize=(int(600*img_rate),int(171*img_rate)))
            # fliped_image = cv2.putText(fliped_image, jangjorim_games.game4.explain, (30, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        if game_num == '5':
            frame_w = cv2.imread(resource_path("jangjorim_games/image/Frame 172.png"), cv2.IMREAD_UNCHANGED)
            frame_w = cv2.resize(frame_w, dsize=(int(600*img_rate),int(171*img_rate)))

        fliped_image = image_overlapping(fliped_image, frame_w, (int(210*img_rate),int(105*img_rate)))


        cv2.namedWindow('posenet')
        cv2.moveWindow('posenet', 40, 30)
        ret, jpeg = cv2.imencode('.jpg', fliped_image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # cv2.imshow('posenet', fliped_image)
        if cv2.waitKey(1) & 0xFF == ord('0'):
            break
    


    time_start = time.time()
    score = 0

    ball_is_here = b'F' # 볼이 여기에 있음을 표현
    ball_is_gone = b'F' # 볼이 떠났음을 표현

    # 중간에 웜홀 이미지 추가
    

    while True:
        while True:
            try:
                input_image, display_image, output_scale = posenet.read_cap(
                    cap, scale_factor=args.scale_factor, output_stride=output_stride)
                break
            except:
                cap = cv2.VideoCapture(args.cam_id)
                cap.set(3, args.cam_width)
                cap.set(4, args.cam_height)
                time.sleep(0.1)
                pass
        

        display_iamge = cv2.resize(display_image, dsize = (0,0), fx = rate, fy = rate)
        display_image = display_image[int(display_image.shape[0]/2)-240:int(display_image.shape[0]/2)+240 ,
                                    int(display_image.shape[1]/2)-320:int(display_image.shape[1]/2)+320]
        

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

        display_image = cv2.resize(display_image, dsize=(640,480))

        # Game Part ( modify only here !!!!!!! ) #####################################
        

        ######################################################
        # Game 1 (Catching star) #############################
        ######################################################
        if game_num == '1':
            # random target generate 
            target_image, target = jangjorim_games.game1.random_target(display_image, target, "star", "origin")

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, touched = jangjorim_games.game1.touch_target(target_image,
                                                            keypoint_coords[0][9:11].astype(np.int32),
                                                            keypoint_coords[0][7:9].astype(np.int32),
                                                            target,
                                                            args.draw_skel)
            
            goal_score = 10
            goal_time = 30

            if touched:
                score += 1
                playsound(sound_mario,block=False)

            fliped_image = cv2.flip(hand_image, 1)
            

            #score board frame
            frame_score = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_score = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_score, (int(20*img_rate),int(20*img_rate)))
            score_txt = "{}/{}".format(score, goal_score)
            fliped_image = text_overlapping(fliped_image, score_txt, (int((20+155/2)*img_rate),int((20+55/2)*img_rate)), 65 ,(250,255,0))
            # fliped_image = cv2.putText(fliped_image, "score : {}/{}".format(score, goal_score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # game name frame
            frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 144.png"), cv2.IMREAD_UNCHANGED)
            frame_game = cv2.resize(frame_game, dsize=(int(138*img_rate),int(34*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_game, (int(546*img_rate),int(20*img_rate)))
            # fliped_image = cv2.putText(fliped_image, game_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # time board frame
            frame_time = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_time = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_time, (int(20*img_rate),int(625*img_rate)))
            time_txt = "{}:{}".format(int(goal_time-time.time()+time_start), int((goal_time-time.time()+time_start)*100)%100)
            fliped_image = text_overlapping(fliped_image, time_txt, (int((625+155/2)*img_rate),int((20+50/2)*img_rate)), 65 ,(255,255,255))            
            # fliped_image = cv2.putText(fliped_image, "{:.2f}".format(goal_time-time.time()+time_start) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if score == goal_score:
                score_txt = "{}/{}".format(score, goal_score)
                time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
                while True:
                    while True:
                        try:
                            input_image, display_image, output_scale = posenet.read_cap(
                                cap, scale_factor=args.scale_factor, output_stride=output_stride)
                            break
                        except:
                            cap = cv2.VideoCapture(args.cam_id)
                            cap.set(3, args.cam_width)
                            cap.set(4, args.cam_height)
                            time.sleep(0.1)
                            pass

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_iamge = cv2.resize(fliped_image, dsize = (0,0), fx = rate, fy = rate)
                    fliped_image = fliped_image[0:480 ,0:640]
                    fliped_image = cv2.resize(fliped_image, dsize=(640,480))
                    
                    # success image frame
                    frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 186.png"), cv2.IMREAD_UNCHANGED)
                    frame_game = cv2.resize(frame_game, dsize=(int(442),int(303)))
                    fliped_image = image_overlapping(fliped_image, frame_game, (int(80),int(99)))
                    # score
                    fliped_image = text_overlapping(fliped_image, score_txt, (int(325),int(272)), 35 ,(255,255,255) , False)
                    # time
                    fliped_image = text_overlapping(fliped_image, time_txt, (int(325),int(305)), 35 ,(255,255,255), False)
                    
                    # fliped_image = cv2.putText(fliped_image, "Success", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    # fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    ret, jpeg = cv2.imencode('.jpg', fliped_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return
            elif goal_time < time.time()-time_start:
                score_txt = "{}/{}".format(score, goal_score)
                time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
                while True:
                    while(True):
                        try:
                            input_image, display_image, output_scale = posenet.read_cap(
                                cap, scale_factor=args.scale_factor, output_stride=output_stride)
                            break
                        except:
                            cap = cv2.VideoCapture(args.cam_id)
                            cap.set(3, args.cam_width)
                            cap.set(4, args.cam_height)
                            time.sleep(0.1)
                            pass

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_iamge = cv2.resize(fliped_image, dsize = (0,0), fx = rate, fy = rate)
                    fliped_image = fliped_image[0:480 ,0:640]
                    fliped_image = cv2.resize(fliped_image, dsize=(640,480))

                    # failure image frame
                    frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 187.png"), cv2.IMREAD_UNCHANGED)
                    frame_game = cv2.resize(frame_game, dsize=(int(442),int(303)))
                    fliped_image = image_overlapping(fliped_image, frame_game, (int(80),int(99)))
                    # score
                    fliped_image = text_overlapping(fliped_image, score_txt, (int(325),int(272)), 32 ,(255,255,255) , False)
                    # time
                    fliped_image = text_overlapping(fliped_image, time_txt, (int(325),int(305)), 32 ,(255,255,255), False)
                    # fliped_image = cv2.putText(fliped_image, "Failed", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    # fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    ret, jpeg = cv2.imencode('.jpg', fliped_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # cv2.imshow('posenet', fliped_image)
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
            target_image, target = jangjorim_games.game2.random_target(display_image, target, "cat")

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, touched = jangjorim_games.game2.touch_target(target_image,
                                                            keypoint_coords[0][9:11].astype(np.int32),
                                                            keypoint_coords[0][7:9].astype(np.int32),
                                                            target,
                                                            args.draw_skel)
            
            if touched:
                score += 1
                playsound(sound_mario,block=False)
                
            game_name = "Game2"

            fliped_image = cv2.flip(hand_image, 1)
            fliped_image = cv2.resize(fliped_image, dsize=(640,480))

            #score board frame
            frame_score = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_score = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_score, (int(20*img_rate),int(20*img_rate)))
            score_txt = "{}/{}".format(score, jangjorim_games.game2.num_of_target)
            fliped_image = text_overlapping(fliped_image, score_txt, (int((20+155/2)*img_rate),int((20+55/2)*img_rate)), 65 ,(250,255,0))
            # fliped_image = cv2.putText(fliped_image, "score : "+str(score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # game name frame
            frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 146.png"), cv2.IMREAD_UNCHANGED)
            frame_game = cv2.resize(frame_game, dsize=(int(138*img_rate),int(34*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_game, (int(546*img_rate),int(20*img_rate)))
            # fliped_image = cv2.putText(fliped_image, game_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)

            # time board frame
            frame_time = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_time = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_time, (int(20*img_rate),int(625*img_rate)))
            time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
            fliped_image = text_overlapping(fliped_image, time_txt, (int((625+155/2)*img_rate),int((20+50/2)*img_rate)), 65 ,(255,255,255))

            if score == jangjorim_games.game2.num_of_target:
                success_time = "Time : " + str(round(time.time() - time_start,2))
                score_txt = "{}/{}".format(score, jangjorim_games.game2.num_of_target)
                time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
                while True:
                    while(True):
                        try:
                            input_image, display_image, output_scale = posenet.read_cap(
                                cap, scale_factor=args.scale_factor, output_stride=output_stride)
                            break
                        except:
                            cap = cv2.VideoCapture(args.cam_id)
                            cap.set(3, args.cam_width)
                            cap.set(4, args.cam_height)
                            time.sleep(0.1)
                            pass

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_iamge = cv2.resize(fliped_image, dsize = (0,0), fx = rate, fy = rate)
                    fliped_image = fliped_image[0:480 ,0:640]
                    fliped_image = cv2.resize(fliped_image, dsize=(640,480))

                    # success image frame
                    frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 186.png"), cv2.IMREAD_UNCHANGED)
                    frame_game = cv2.resize(frame_game, dsize=(int(442),int(303)))
                    fliped_image = image_overlapping(fliped_image, frame_game, (int(80),int(99)))
                    # score
                    fliped_image = text_overlapping(fliped_image, score_txt, (int(325),int(272)), 32 ,(255,255,255), False)
                    # time
                    fliped_image = text_overlapping(fliped_image, time_txt, (int(325),int(305)), 32 ,(255,255,255), False)
                    # fliped_image = cv2.putText(fliped_image, "Success", (250, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    # fliped_image = cv2.putText(fliped_image, success_time , (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    # fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    ret, jpeg = cv2.imencode('.jpg', fliped_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # cv2.imshow('posenet', fliped_image)
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
            frame_bad = cv2.imread(resource_path("jangjorim_games/image/Frame 154.png"), cv2.IMREAD_UNCHANGED)
            frame_bad = cv2.resize(frame_bad, dsize=(int(84*img_rate),int(30*img_rate)))
            frame_great = cv2.imread(resource_path("jangjorim_games/image/Frame 153.png"), cv2.IMREAD_UNCHANGED)
            frame_great = cv2.resize(frame_great, dsize=(int(135*img_rate),int(31*img_rate)))
            frame_perf = cv2.imread(resource_path("jangjorim_games/image/Frame 155.png"), cv2.IMREAD_UNCHANGED)
            frame_perf = cv2.resize(frame_perf, dsize=(int(189*img_rate),int(31*img_rate)))

            height = display_image.shape[0]
            hit_range = [
                ("Perfect", int(height/2-height/20), int(height/2+height/20), frame_perf),
                ("Great", int(height/2-height/8), int(height/2+height/8), frame_great),
                ("Bad", int(height/2-height/2), int(height/2+height/2), frame_bad)
            ]

            # random target generate 
            target_image, target = jangjorim_games.game3.random_target(display_image, target, "volleyball", "original")

            # index 9 = left wrist, index 10 = right wrist
            # check if hand close to target    
            hand_image, target, touched, scored, touch_img = jangjorim_games.game3.touch_target(target_image,
                                                            keypoint_coords[0][9:11].astype(np.int32),
                                                            keypoint_coords[0][7:9].astype(np.int32),
                                                            target,
                                                            hit_range,
                                                            touched,
                                                            touch_img,
                                                            args.draw_skel)
            if touched == None:
                score = 0
            else:
                score += scored
                # playsound(sound_mario,block=False)


            game_name = "Volley ball challenge"

            fliped_image = cv2.flip(hand_image, 1)
            fliped_image = cv2.resize(fliped_image, dsize=(640,480))

            # 기준선 
            fliped_image = cv2.line(fliped_image, (50,int(height/2)),(100,int(height/2)),(0,0,255),6)
            
            # Score function
            #score board frame
            frame_score = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_score = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_score, (int(20*img_rate),int(20*img_rate)))
            score_txt = "{}".format(score)
            fliped_image = text_overlapping(fliped_image, score_txt, (int((20+155/2)*img_rate),int((20+55/2)*img_rate)), 65 ,(250,255,0))
            
            # 결과 이미지
            if touch_img is not None:
                fliped_image = image_overlapping(fliped_image, touch_img, (int(285*img_rate),int(415*img_rate)))

            
            # fliped_image = cv2.putText(fliped_image, touched, (300, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            
            # game name frame
            frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 151.png"), cv2.IMREAD_UNCHANGED)
            frame_game = cv2.resize(frame_game, dsize=(int(138*img_rate),int(34*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_game, (int(546*img_rate),int(20*img_rate)))
            # fliped_image = cv2.putText(fliped_image, game_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
            
            # fliped_image = cv2.putText(fliped_image, "score : "+str(score), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # time board frame
            frame_time = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_time = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_time, (int(20*img_rate),int(625*img_rate)))
            time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
            fliped_image = text_overlapping(fliped_image, time_txt, (int((625+155/2)*img_rate),int((20+50/2)*img_rate)), 65 ,(255,255,255))

        ######################################################  
        # Game 3 #############################################
        ######################################################


        ######################################################
        # Game 4 (English touching) ##########################
        ######################################################
        if game_num == '4':
            # random target generate 
            target_image, target, part = jangjorim_games.game4.random_target(display_image, target, part, "design", "origin")

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
                playsound(sound_mario,block=False)

            game_name = "Game4"

            fliped_image = cv2.flip(hand_image, 1)
            fliped_image = cv2.resize(fliped_image, dsize=(640,480))
            # if target is not None:
                # fliped_image = cv2.putText(fliped_image, part, (fliped_image.shape[1]-target[0]-30,target[1]+8), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),2)
            # fliped_image = cv2.putText(fliped_image, "score : {}/{}".format(score, goal_score), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            

            # socre frame
            frame_score = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_score = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_score, (int(20*img_rate),int(20*img_rate)))
            score_txt = "{}/{}".format(score, goal_score)
            fliped_image = text_overlapping(fliped_image, score_txt, (int((20+155/2)*img_rate),int((20+55/2)*img_rate)), 65 ,(250,255,0))


            # game name frame
            frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 156.png"), cv2.IMREAD_UNCHANGED)
            frame_game = cv2.resize(frame_game, dsize=(int(138*img_rate),int(34*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_game, (int(546*img_rate),int(20*img_rate)))
            # fliped_image = cv2.putText(fliped_image, game_name, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            

            # time frame
            frame_time = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_time = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_time, (int(20*img_rate),int(625*img_rate)))
            time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
            fliped_image = text_overlapping(fliped_image, time_txt, (int((625+155/2)*img_rate),int((20+50/2)*img_rate)), 65 ,(255,255,255))
            # fliped_image = cv2.putText(fliped_image, "{:.2f}".format(goal_time-time.time()+time_start) , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            if score == goal_score:
                score_txt = "{}/{}".format(score, goal_score)
                time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
                while True:
                    while(True):
                        try:
                            input_image, display_image, output_scale = posenet.read_cap(
                                cap, scale_factor=args.scale_factor, output_stride=output_stride)
                            break
                        except:
                            cap = cv2.VideoCapture(args.cam_id)
                            cap.set(3, args.cam_width)
                            cap.set(4, args.cam_height)
                            time.sleep(0.1)
                            pass

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_iamge = cv2.resize(fliped_image, dsize = (0,0), fx = rate, fy = rate)
                    fliped_image = fliped_image[0:480 ,0:640]
                    fliped_image = cv2.resize(fliped_image, dsize=(640,480))

                    # success image frame
                    frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 186.png"), cv2.IMREAD_UNCHANGED)
                    frame_game = cv2.resize(frame_game, dsize=(int(442),int(303)))
                    fliped_image = image_overlapping(fliped_image, frame_game, (int(80),int(99)))
                    # score
                    fliped_image = text_overlapping(fliped_image, score_txt, (int(325),int(272)), 32 ,(255,255,255), False)
                    # time
                    fliped_image = text_overlapping(fliped_image, time_txt, (int(325),int(305)), 32 ,(255,255,255), False)
                    # fliped_image = cv2.putText(fliped_image, "Success", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    # fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    ret, jpeg = cv2.imencode('.jpg', fliped_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # cv2.imshow('posenet', fliped_image)
                    if cv2.waitKey(1) & 0xFF == ord('0'):
                        return
            elif goal_time < time.time()-time_start:
                score_txt = "{}/{}".format(score, goal_score)
                time_txt = "{}:{}".format(int(time.time()-time_start), int((time.time()-time_start)*100)%100)
                while True:
                    while(True):
                        try:
                            input_image, display_image, output_scale = posenet.read_cap(
                                cap, scale_factor=args.scale_factor, output_stride=output_stride)
                            break
                        except:
                            cap = cv2.VideoCapture(args.cam_id)
                            cap.set(3, args.cam_width)
                            cap.set(4, args.cam_height)
                            time.sleep(0.1)
                            pass

                    fliped_image = cv2.flip(display_image, 1)
                    fliped_iamge = cv2.resize(fliped_image, dsize = (0,0), fx = rate, fy = rate)
                    fliped_image = fliped_image[0:480 ,0:640]
                    fliped_image = cv2.resize(fliped_image, dsize=(640,480))
                    # success image frame
                    frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 187.png"), cv2.IMREAD_UNCHANGED)
                    frame_game = cv2.resize(frame_game, dsize=(int(442),int(303)))
                    fliped_image = image_overlapping(fliped_image, frame_game, (int(80),int(99)))
                    # score
                    fliped_image = text_overlapping(fliped_image, score_txt, (int(325),int(272)), 32 ,(255,255,255), False)
                    # time
                    fliped_image = text_overlapping(fliped_image, time_txt, (int(325),int(305)), 32 ,(255,255,255), False)
                    # fliped_image = cv2.putText(fliped_image, "Failed", (250, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
                    # fliped_image = cv2.putText(fliped_image, "Press 0 to go to Game Menu" , (40, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                    ret, jpeg = cv2.imencode('.jpg', fliped_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # cv2.imshow('posenet', fliped_image)
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
            oppos_score = b'F'
            ball_is_gone = b'F'
            if ball_is_here == b'T':

              # random target generate 
              display_image, target, gone = jangjorim_games.game5.random_target(
                display_image, target, time.time()-time_start,"soccerball", "original")
              time_start = time.time()

            #   if gone is None:
                
            #     # score -= 1
              if gone:
                ball_is_gone = b'T'
              elif gone is not None:
                oppos_score = b'T'
                another_score += 1
                ball_is_gone = b'F'
              

              # index 9 = left wrist, index 10 = right wrist
              # check if hand close to target    
              display_image, target, scored= jangjorim_games.game5.touch_target(display_image,
                                                              keypoint_coords[0][9:11].astype(np.int32),
                                                              keypoint_coords[0][7:9].astype(np.int32),
                                                              target,
                                                              args.draw_skel)

              # score 계산 ( 뒷빡 없음 )                                            
            #   score += scored
              if scored == 1:
                playsound(sound_mario,block=False)
              if score < 0 :
                score = 0

            game_name = "Volley ball challenge"

            fliped_image = cv2.flip(display_image, 1)
            fliped_image = cv2.resize(fliped_image, dsize=(640,480))

            
                    # image server로 보낼 때 downsampling
            # down_img = cv2.resize(fliped_image, dsize=(0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)            
            # fliped_image = cv2.resize(fliped_image, dsize=(int(args.cam_width*2/3), int(args.cam_height*2/3)))

            # cv2. imencode(ext, img [, params])
            # encode_param의 형식으로 frame을 jpg로 이미지를 인코딩한다.
            result, frame = cv2.imencode('.jpg', fliped_image, encode_param)
            # frame을 String 형태로 변환
            data = np.array(frame)
            stringData = data.tostring()

            # 공이 있는지 없는지 판단하는 거 뒤에다가 붙여줌
            stringData += oppos_score + ball_is_here + ball_is_gone
        
            #서버에 데이터 전송
            #(str(len(stringData))).encode().ljust(16)
            client_socket.sendall((str(len(stringData))).encode().ljust(16) + stringData)

            print("get_data")
            # 서버에서 통합된 영상 받아서 화면에 내보내기
            length = recvall(client_socket, 16)
            stringData = recvall(client_socket, int(length))

            # 공이 있는지 없는지 저장해주는 것
            ball_is_here = stringData[-1:]
            my_scored = stringData[-2:-1]
            if my_scored == b'T':
                my_score += 1
            stringData = stringData[:-2]

            data = np.fromstring(stringData, dtype = 'uint8')
            #data를 디코딩한다.
            frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

            # 높이를 480으로 고정하기 위한 작업
            h,w,_ = fliped_image.shape
            # image upsampling
            mid_img = cv2.imread(resource_path("jangjorim_games/image/Frame 167.png"), cv2.IMREAD_UNCHANGED)
            mid_img = mid_img[:,:,:3]
            mid_img = cv2.resize(mid_img, dsize = (mid_img.shape[1],h))
            frame = cv2.resize(frame, dsize=(w,h), interpolation=cv2.INTER_CUBIC)



            # Score frame (my score)
            frame_score = cv2.imread(resource_path("jangjorim_games/image/Frame 138.png"), cv2.IMREAD_UNCHANGED)
            frame_score = cv2.resize(frame_score, dsize=(int(155*img_rate),int(65*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_score, (int(20*img_rate),int(625*img_rate)))
            score_txt = "{}".format(my_score)
            fliped_image = text_overlapping(fliped_image, score_txt, (int((625+155/2)*img_rate),int((20+55/2)*img_rate)), 65 ,(250,255,0))

            # Score frame (oppose score)
            frame = image_overlapping(frame, frame_score, (int(20*img_rate),int(20*img_rate)))
            score_txt = "{}".format(another_score)
            frame = text_overlapping(frame, score_txt, (int((20+155/2)*img_rate),int((20+55/2)*img_rate)), 65 ,(250,255,0))

            # fliped_image = cv2.putText(fliped_image, "score : "+str(score), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # game name frame
            frame_game = cv2.imread(resource_path("jangjorim_games/image/Frame 156.png"), cv2.IMREAD_UNCHANGED)
            frame_game = cv2.resize(frame_game, dsize=(int(138*img_rate),int(34*img_rate)))
            fliped_image = image_overlapping(fliped_image, frame_game, (int(546*img_rate),int(20*img_rate)))
            frame = image_overlapping(frame, frame_game, (int(546*img_rate),int(642*img_rate)))


            final = cv2.hconcat([fliped_image, mid_img])
            fliped_image = cv2.hconcat([final, frame])

        ######################################################  
        # Game 5 #############################################
        ######################################################



        ################################################################################

        
        # cv2.namedWindow('posenet', cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty('posenet', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        # cv2.imshow('posenet', fliped_image)
        cv2.namedWindow('posenet')
        cv2.moveWindow('posenet', 40, 30)
        ret, jpeg = cv2.imencode('.jpg', fliped_image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # cv2.imshow('posenet', fliped_image)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('0'):
            return

    print('Average FPS: ', frame_count / (time.time() - start))


def start_loop():
    print(2)
    try:
        print(3)
        # sound_park = resource_path("jangjorim_games/sound/park.mp3")
        # sound_park = sound_park.replace(" ", "%20")
        # playsound(sound_park, block= False)
        skel_draw = input("Do you want to draw skeleton? Y/N")

        if skel_draw == 'Y':
            args.draw_skel = True
        else:
            args.draw_skel = False

        # HOST = input("Insert IP please : ")
        HOST = "127.0.0.1"

        # jumple 시작 화면 
        start_jumple = time.time()
        start_img = cv2.imread(resource_path("jangjorim_games/image/Group 46.png"), cv2.IMREAD_UNCHANGED)
        
        blank_img = np.zeros((480,640,3),np.uint8)
        blank_image = Image.fromarray(cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB))
        blank_img = cv2.cvtColor(np.array(blank_image), cv2.COLOR_RGB2BGR)
        blank_img = image_overlapping(blank_img, start_img, (177,177))
        
        cv2.namedWindow('posenet')
        cv2.moveWindow('posenet', 40, 30)
        ret, jpeg = cv2.imencode('.jpg', blank_image)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        # cv2.imshow('posenet', blank_img)

        key = cv2.waitKey(4000)
        time.sleep(0.1)
            

        while True:

            blank_img = np.zeros((480,640,3),np.uint8)

            pill_image = Image.fromarray(cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pill_image)

            draw.text((40, 40), "Press the number to start playing", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 105), "1 : Catch the stars", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 150), "2 : Pop balloons", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 195), "3 : Volleyball(single player)", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 240), "4 : Learn your body parts", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 285), "5 : Volleyball(2 player)", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 330), "6 : Game credits", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            draw.text((40, 375), "Q : Quit game", font=ImageFont.truetype(font_dir, 36), fill=(255, 255, 255))
            
            blank_img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR) #맥


            # blank_img = cv2.putText(blank_img, "Press button for start game",(30,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            # blank_img = cv2.putText(blank_img, "1 : Catching Stars",(30,105), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            # blank_img = cv2.putText(blank_img, "2 : Pop Ballon",(30,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            # blank_img = cv2.putText(blank_img, "3 : Volley Ball Alone",(30,195), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            # blank_img = cv2.putText(blank_img, "4 : English ",(30,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            # blank_img = cv2.putText(blank_img, "5 : volley together",(30,285), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            # blank_img = cv2.putText(blank_img, "0 : Quit Game",(30,330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
            
            time.sleep(0.1)
            cv2.namedWindow('posenet')
            cv2.moveWindow('posenet', 40, 30)
            ret, jpeg = cv2.imencode('.jpg', blank_image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            # cv2.imshow('posenet', blank_img)

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
                if key == ord('6'):
                    
                    blank_img = np.zeros((480,640,3),np.uint8)
                    pill_image = Image.fromarray(cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(pill_image)

                    draw.text((40, 40), "Game credits", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 110), "Produced by Shinjangjorim", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 145), "Jae kyoung Cho,Sang won Im, ", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 180), "Jae young Jang, Soo hyun Shin", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 250), "Game Programmer: Jae kyoung Cho", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 285), "jackyoung96@snu.ac.kr", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 355), "Game Designer: Soo hyun Shin", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    draw.text((40, 390), "soohyunshin@snu.ac.kr", font=ImageFont.truetype(font_dir, 34), fill=(255, 255, 255))
                    

                    blank_img = cv2.cvtColor(np.array(pill_image), cv2.COLOR_RGB2BGR)
                    time.sleep(0.1)
                    cv2.namedWindow('posenet')
                    cv2.moveWindow('posenet', 40, 30)
                    ret, jpeg = cv2.imencode('.jpg', blank_image)
                    frame = jpeg.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
                    # cv2.imshow('posenet', blank_img)
                    while True:
                        key = cv2.waitKey()
                        if key is not None:
                            raise getOutOfLoop


                if key == ord('q'):
                    raise getOutOfLoop

            main_loop(game_type)
    
    except getOutOfLoop:
        pass