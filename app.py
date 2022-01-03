from flask import Flask, Response
import cv2
from Flask.jumple_student import start_loop, image_overlapping

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




app = Flask(__name__)
video = cv2.VideoCapture(0)
font_dir = resource_path("jangjorim_games/font/VT323-Regular.ttf")
HOST = "127.0.0.1"
PORT = 5036


@app.route('/')
def index():
    return "Default Message"

def gen():
    start_jumple = time.time()
    start_img = cv2.imread(resource_path("jangjorim_games/image/Group 46.png"), cv2.IMREAD_UNCHANGED)
    
    blank_img = np.zeros((480,640,3),np.uint8)
    blank_image = Image.fromarray(cv2.cvtColor(blank_img, cv2.COLOR_BGR2RGB))
    blank_img = cv2.cvtColor(np.array(blank_image), cv2.COLOR_RGB2BGR)
    blank_img = image_overlapping(blank_img, start_img, (177,177))
    
    # cv2.namedWindow('posenet')
    # cv2.moveWindow('posenet', 40, 30)
    ret, jpeg = cv2.imencode('.jpg', np.array(blank_image))
    frame = jpeg.tobytes()
    yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    Response((b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n'), mimetype='multipart/x-mixed-replace; boundary=frame')
    # cv2.imshow('posenet', blank_img)
    print(1)
    key = cv2.waitKey(4000)
    print(2)
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
        ret, jpeg = cv2.imencode('.jpg', np.array(blank_image))
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
                ret, jpeg = cv2.imencode('.jpg', np.array(blank_image))
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

        # main_loop(game_type)

    # while True:
    #     print(2)
    #     success, image = video.read()
    #     ret, jpeg = cv2.imencode('.jpg', image)
    #     frame = jpeg.tobytes()
    #     yield (b'--frame\r\n'
    #            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    global video
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)