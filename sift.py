import cv2
import matplotlib.pyplot as plt

image1 = cv2.imread('C:\\Users\\ravik\\Downloads\\lib1.jpg') 
image2 = cv2.imread('C:\\Users\\ravik\\Downloads\\lib2.jpg') 

gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(gray_image2, None)
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(descriptors1, descriptors2, k=2)
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:  
        good_matches.append(m)
output_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Matches between Image 1 and Image 2 - {len(good_matches)} Good Matches")
plt.axis('off')
plt.show()
