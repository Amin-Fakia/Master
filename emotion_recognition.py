from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread("WIN_20230101_18_23_37_Pro.jpg")
# plt.imshow(img1[:,:,::-1])
# plt.show()

result = DeepFace.analyze(img1,actions=['emotion'])

print(result)