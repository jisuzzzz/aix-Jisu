import cv2
import numpy as np
from matplotlib import pyplot as plt

face_img = cv2.imread("얼굴사진.png")
eye_img = cv2.imread("눈사진.png")

face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
eye_img = cv2.cvtColor(eye_img, cv2.COLOR_BGR2RGB)

plt.imshow(face_img)
plt.show()
plt.imshow(eye_img)
plt.show()

face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
eye_gray = cv2.cvtColor(eye_img, cv2.COLOR_BGR2GRAY)
eye_reverse = cv2.flip(eye_gray, 1)


find_left = cv2.matchTemplate(face_gray, eye_gray, cv2.TM_CCOEFF_NORMED)
min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(find_left)
find_right = cv2.matchTemplate(face_gray, eye_reverse, cv2.TM_CCOEFF_NORMED)
min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(find_right)

point_left = max_loc1
point_right = max_loc2


print("왼쪽 눈 위치: ", point_left)
print("오른쪽 눈 위치: ", point_right)
w, h = eye_gray.shape[::-1]

cv2.rectangle(face_img, point_left, (point_left[0] + w, point_left[1] + h), (0,0,255), 2)
cv2.rectangle(face_img, point_right, (point_right[0] + w, point_right[1] + h), (0,0,255), 2)
plt.imshow(face_img)
plt.show()

image_1 = cv2.imread('3.png')
image_2 = cv2.imread('4.png')
image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
gray_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)
gray_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray_1,None)
kp2, des2 = orb.detectAndCompute(gray_2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)

res = cv2.drawMatches(image_1, kp1, image_2, kp2, matches[:100],None,flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.figure(figsize=(18,18))
plt.imshow(res)
plt.show()

img = cv2.imread('얼굴사진.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

noise_img = img.copy()
noise_img = np.clip((img / 255 + np.random.normal(scale=0.1, size=img.shape)) * 255, 0, 255).astype('uint8')

filtered_img = cv2.GaussianBlur(noise_img, (5, 5), 0)

plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.title('original image')
plt.imshow(img)
plt.subplot(1,3,2)
plt.title('image with noise')
plt.imshow(noise_img)
plt.subplot(1,3,3)
plt.title('filtered image')
plt.imshow(filtered_img)
plt.show()

np.random.seed(42)
noise_img2 = img.copy()
N = 100000
x = np.random.randint(img.shape[0], size=N)
y = np.random.randint(img.shape[1], size=N)
noise_img2[x, y] = 0

kernal_size = 7
filtered_img2 = cv2.medianBlur(noise_img2, kernal_size)

plt.figure(figsize=(12,12))
plt.subplot(1,3,1)
plt.title('original image')
plt.imshow(img)
plt.subplot(1,3,2)
plt.title('image with noise')
plt.imshow(noise_img2)
plt.subplot(1,3,3)
plt.title('filtered image')
plt.imshow(filtered_img2)
plt.show()

img = cv2.imread('얼굴사진.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel_img = cv2.Sobel(img,-1,1,0)


plt.imshow(sobel_img, cmap='gray')
plt.show()


img = cv2.imread('feature_extracting.jpeg')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = np.float32(gray_img)

dst = cv2.cornerHarris(gray_img, 2, 5, 0.15)
dst = cv2.dilate(dst,None)
img[dst>0.01*dst.max()]=[0,0,255]

plt.imshow(img)
plt.show()

imgL = cv2.imread('DM_left_image.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('DM_right_image.png', cv2.IMREAD_GRAYSCALE)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()