import cv2
import random
import math
import numpy as np
from jangjorim_client import resource_path

# constants
threshold = 50
interpol = 0.7
velocity = 0
num_of_target = 15

# game explanation
explain = "Pop all ballons!!!!"

# target shape
img_rate = 0.8
ballon_img = cv2.imread(resource_path("jangjorim_games/image/Frame 173.png"), cv2.IMREAD_UNCHANGED)
ballon_img = cv2.resize(ballon_img, dsize=(int(56*img_rate),int(94*img_rate)))

# target shape
bubble_size = 0.25
bubble_img = cv2.imread(resource_path("jangjorim_games/image/bubble.png"), cv2.IMREAD_UNCHANGED)
bubble_img = cv2.resize(bubble_img, dsize=(0,0), fx=bubble_size,fy=bubble_size)

# target shape (cat version)
img_rate = 0.8
cat_img = cv2.imread(resource_path("jangjorim_games/image/Frame 147.png"), cv2.IMREAD_UNCHANGED)
cat_img = cv2.resize(cat_img, dsize=(int(95*img_rate),int(211*img_rate)))




def distance(hand, target):
  distance = []
  for i in range(2):
    distance.append(math.sqrt((hand[i][0]-target[0])**2 + (hand[i][1]-target[1])**2))
  
  return min(distance[0], distance[1])

def touch_target(img, hand, elbow , target, draw_hand):
  touched = False
  left_hand = (((1+interpol)*hand[0][1]-interpol*elbow[0][1]).astype(np.int32), ((1+interpol)*hand[0][0]-interpol*elbow[0][0]).astype(np.int32))
  right_hand = (((1+interpol)*hand[1][1]-interpol*elbow[1][1]).astype(np.int32), ((1+interpol)*hand[1][0]-interpol*elbow[1][0]).astype(np.int32))
  if draw_hand:
    img = cv2.circle(img, left_hand, radius = 15, color=(0,255,0), thickness=-1)
    img = cv2.circle(img, right_hand, radius = 15, color=(0,255,0), thickness=-1)

  if not target is None:
    for i in range(len(target)):
      if distance([left_hand,right_hand], target[i]) < threshold:
        del target[i]
        touched = True
        break
      

  return img, target, touched

def random_target(img, target, target_type="ballon"):
  if target is None:
    target = []
    for i in range(num_of_target):
      target.append((random.randint(30,img.shape[1]-30), random.randint(30,img.shape[0]-30)))

  if target_type is "ballon":
    target_img = ballon_img
  if target_type is "cat":
    target_img = cat_img
  if target_type is "bubble":
    target_img = bubble_img

  for i in target:
    img = image_overlapping(img, target_img, i)

  return img, target            


def image_overlapping(screen, img, pos):
    x_size, y_size, _ = img.shape
    x1, x2 = pos[1]-int(x_size/2), pos[1]+x_size-int(x_size/2)
    y1, y2 = pos[0]-int(y_size/2), pos[0]+y_size-int(y_size/2)
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

# def image_overlapping(screen, img, pos):

#   y_size, x_size, _ = img.shape
#   x1, x2 = pos[0]-int(x_size/2), pos[0]+x_size-int(x_size/2)
#   y1, y2 = pos[1]-int(y_size/2), pos[1]+y_size-int(y_size/2)
#   if y2 >= screen.shape[0]:
#     y2 = screen.shape[0]-1
#     y1 = y2 - y_size
#   if y1 < 0:
#     y1 = 0
#     y2 = y1 + y_size
#   if x2 >= screen.shape[1]:
#     x2 = screen.shape[1]-1
#     x1 = x2 - x_size
#   if x1 < 0:
#     x1 = 0
#     x2 = x1 + x_size

#   alpha_img = img[:,:,3]/255.0
#   alpha_screen = 1- alpha_img

#   color = (255,255,255)
#   for c in range(3):
#     screen[y1:y2, x1:x2, c] = alpha_img * img[:,:,c] * color[c]/255.0 + alpha_screen * screen[y1:y2, x1:x2, c]

#   return screen