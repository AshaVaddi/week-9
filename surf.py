import cv2
import matplotlib.pyplot as plt
image = cv2.imread('C:\\Users\\ravik\\Downloads\\lib1.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
surf = cv2.xfeatures2d.SURF_create()
keypoints, descriptors = surf.detectAndCompute(gray_image, None)
output_image = cv2.drawKeypoints(image, keypoints, None, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title(f"Image with {len(keypoints)} SURF Keypoints")
plt.axis('off')
plt.show()
