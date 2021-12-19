import cv2
import numpy as np

MIN_MATCH_COUNT = 5

img1 = cv2.imread("knjiga_naslovnica.png", cv2.IMREAD_GRAYSCALE)  # query image
img2 = cv2.imread("knjige.png", cv2.IMREAD_GRAYSCALE)  # train image

sift = cv2.SIFT_create()

kp1, desc1 = sift.detectAndCompute(img1, None)

img1_keypoints = cv2.drawKeypoints(
    img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

kp2, desc2 = sift.detectAndCompute(img2, None)

# img2_keypoints = cv2.drawKeypoints(
#     img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("keys", img1_keypoints)
# cv2.imshow("keys2", img2_keypoints)

matcher = cv2.BFMatcher()
matches = matcher.knnMatch(desc1, desc2, k=2)

good_matches = []

for m, n in matches:
    if m.distance < 0.7*n.distance:
        good_matches.append(m)

# img_matches = cv2.drawMatchesKnn(
#     img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

if len(good_matches) > MIN_MATCH_COUNT:
    src_points = np.float32(
        [kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dest_points = np.float32(
        [kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(src_points, dest_points, cv2.RANSAC, 5.0)

    h, w = img1.shape
    pts = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1],
                      [0, h - 1]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, M)

    img_detection = cv2.polylines(img2, [np.int32(dst)], True, 255, 3)

cv2.imshow("slika", img_detection)
cv2.waitKey()
cv2.destroyAllWindows()
