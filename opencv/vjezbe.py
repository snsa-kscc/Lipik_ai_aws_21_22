import cv2

# img = cv2.imread('kamera1.jpg', cv2.IMREAD_COLOR)

# bigger_img = cv2.resize(img, None, fx=2, fy=2)


# cv2.imshow('image', img)
# cv2.imshow('image bigger', bigger_img)

# cv2.waitKey(7000)
# cv2.destroyAllWindows()


slika = cv2.imread('kamera1.jpg', cv2.IMREAD_COLOR)

# slika[10:50, 40:60, :] = [0, 0, 255]

silka = slika[:, :, :2] = 0

cv2.imshow('image', slika)
cv2.waitKey(5000)
cv2.destroyAllWindows()
