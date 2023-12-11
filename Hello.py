import cv2
from google.colab.patches import cv2_imshow
import numpy as np
import dlib
import time
import glob


!wget https://raw.githubusercontent.com/italojs/facial-landmarks-recognition/master/shape_predictor_68_face_landmarks.dat


import os
import requests
import json

# GitHub repository URL
repo_url = "https://api.github.com/repos/96gang96/ImagesDump/git/trees/main?recursive=1"

def download_images_from_github(repo_url):
    # Fetch the tree structure of the repository
    response = requests.get(repo_url)
    if response.status_code != 200:
        print("Failed to fetch repository information.")
        return

    repo_data = response.json()
    files = [item for item in repo_data['tree'] if item['type'] == 'blob' and item['path'].lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Iterate through files and download them
    for file_data in files:
        file_path = file_data['path']
        file_url = f"https://raw.githubusercontent.com/96gang96/ImagesDump/main/{file_path}"

        # Download the file to the current directory
        file_name = os.path.basename(file_path)
        local_file_path = os.path.join(os.getcwd(), file_name)
        response = requests.get(file_url)
        with open(local_file_path, "wb") as f:
            f.write(response.content)
            print(f"Downloaded {file_name} to {local_file_path}")

# Download images from the specified GitHub repository
download_images_from_github(repo_url)
def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

path = '/content/test_cv_imgs/*.*'

img = cv2.imread("/content/WIN_20231019_08_42_38_Pro.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)

for file in glob.glob(path):
  print(file)

for file in glob.glob(path):

  print(file)
  img2 = cv2.imread(file)
  print(img2)
  img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


  detector = dlib.get_frontal_face_detector()
  predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
  height, width, channels = img2.shape
  img2_new_face = np.zeros((height, width, channels), np.uint8)




  # Face 1
  faces = detector(img_gray)
  for face in faces:
      landmarks = predictor(img_gray, face)
      landmarks_points = []
      for n in range(0, 68):
          x = landmarks.part(n).x
          y = landmarks.part(n).y
          landmarks_points.append((x, y))



      points = np.array(landmarks_points, np.int32)
      convexhull = cv2.convexHull(points)
      # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
      cv2.fillConvexPoly(mask, convexhull, 255)

      face_image_1 = cv2.bitwise_and(img, img, mask=mask)

      # Delaunay triangulation
      rect = cv2.boundingRect(convexhull)
      subdiv = cv2.Subdiv2D(rect)
      subdiv.insert(landmarks
