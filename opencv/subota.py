# %%
import cv2
import numpy as np
# %%
slika = cv2.imread('kamera1.jpg', cv2.IMREAD_GRAYSCALE)

ret, treshold = cv2.threshold(slika, 127, 255, cv2.THRESH_BINARY)

cv2.imshow('original', slika)
cv2.imshow('treshold', treshold)
cv2.waitKey(5000)
cv2.destroyAllWindows()
# %%
slika_bgr = cv2.imread('kamera1.jpg')
slika_hsv = cv2.cvtColor(slika_bgr, cv2.COLOR_BGR2HSV)

zelena_donja = np.array([75, 80, 80])
zelena_gornja = np.array([88, 255, 255])

maska = cv2.inRange(slika_hsv, zelena_donja, zelena_gornja)

print(maska.shape)
print(slika_bgr.shape)
filter_slika_bgr = slika_bgr * (maska[:, :, None] // 255)  # ovo treba skužiti

# filter_slika_hsv = cv2.bitwise_and(slika_hsv, slika_hsv, mask=maska)
# filter_slika_bgr = cv2.cvtColor(filter_slika_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('original', slika_bgr)
cv2.imshow('maska', maska)
cv2.imshow('filter', filter_slika_bgr)
cv2.waitKey(9000)
cv2.destroyAllWindows()

# %%
slika = cv2.imread('kamera1.jpg')
zamucena = cv2.blur(slika, (5, 5))

cv2.imshow('original', slika)
cv2.imshow('zamucena', zamucena)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %%
slika = cv2.imread('kamera1.jpg')
zamucena = cv2.GaussianBlur(slika, (5, 5), 0)

cv2.imshow('original', slika)
cv2.imshow('zamucena', zamucena)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %%
slika = cv2.imread('kamera2.jpg')

grayscale = cv2.cvtColor(slika, cv2.COLOR_BGR2GRAY)
slika_zamucena = cv2.GaussianBlur(grayscale, (5, 5), 0)

# ovo mi nije jasno
# nova_slika = cv2.fastNlMeansDenoising(slika_zamucena, None, 10, 10, 7)
# sobel_x_abs = cv2.convertScaleAbs(nova_slika)

sobel_x = cv2.Sobel(slika_zamucena, cv2.CV_64F, 1, 0, ksize=5)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)


cv2.imshow("original", slika)
cv2.imshow("sobel_x", sobel_x_abs)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %%
slika = cv2.imread("kamera2.jpg")
canny = cv2.Canny(slika, 100, 200)

cv2.imshow("original", slika)
cv2.imshow("canny", canny)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% . crtanje linije
slika = np.zeros((500, 500, 3))

p1 = (0, 0)
p2 = (250, 250)
line_color = [0, 255, 0]

cv2.line(slika, p1, p2, line_color)

cv2.imshow("linija", slika)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% crtanje kruznice
slika = np.zeros((500, 500, 3))

cicrle_center = (100, 100)
circle_radius = 100
circle_color = [255, 0, 0]

cv2.circle(slika, cicrle_center, circle_radius, circle_color)

cv2.imshow("krug", slika)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% maskiranje slike
im = cv2.imread('kamera4.png', cv2.IMREAD_GRAYSCALE)

mask = np.zeros_like(im)
height, width = im.shape

x1 = width / 2
y1 = 0
x2 = width
y2 = 0
x3 = width
y3 = height
x4 = width / 2
y4 = height

rectangle_points = np.array(
    [[(x1, y1), (x2, y2), (x3, y3), (x4, y4)]], dtype=np.int32)

mask = cv2.fillPoly(mask, rectangle_points, 255)
masked_image = cv2.bitwise_and(im, mask)

cv2.imshow("original", im)
cv2.imshow("mask", mask)
cv2.imshow("masked_image", masked_image)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %%
