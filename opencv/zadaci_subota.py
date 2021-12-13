# %%
import cv2
import numpy as np
# %% zadatak 1
slika = cv2.imread('kamera1.jpg', cv2.IMREAD_GRAYSCALE)
print(slika.shape)

cv2.imshow('original', slika)
cv2.imwrite('kamera1_crno_bijela.jpg', slika)
cv2.waitKey(5000)
cv2.destroyAllWindows()


# %% zadatak 2
slika2 = cv2.imread('kamera2.jpg')
print(slika2.shape)

resized = cv2.resize(slika2, (650, 1000))

print(resized.shape)
cv2.imshow('resized', resized)
cv2.waitKey(5000)
cv2.destroyAllWindows()


# %% zadatak 3
slika3 = cv2.imread('kamera2_1.jpg')
slika3[273:282:, 143:154:, :] = [0, 255, 255]

cv2.imshow('slika3', slika3)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% zadatak 4
slika4 = cv2.imread('kamera2_1.jpg')
img_hsv = cv2.cvtColor(slika4, cv2.COLOR_BGR2HSV)

img_hsv[273:282:, 143:154:, :] = [30, 255, 255]

img_hsv_to_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('slika4', img_hsv_to_bgr)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% zadatak 5
slika5 = cv2.imread('kamera2.jpg', cv2.IMREAD_GRAYSCALE)

img_mean = slika5.mean()

_, treshold_mean = cv2.threshold(slika5, img_mean, 255, cv2.THRESH_BINARY)

cv2.imshow('slika5', treshold_mean)
cv2.waitKey(5000)
cv2.destroyAllWindows()


# %% zadaatak 6
slika6 = cv2.imread('crveni_semafor.jpg')
slika_hsv = cv2.cvtColor(slika6, cv2.COLOR_BGR2HSV)

crvena_donja = np.array([165, 80, 80])
crvena_gornja = np.array([179, 255, 255])

maska = cv2.inRange(slika_hsv, crvena_donja, crvena_gornja)

filter_slika_hsv = cv2.bitwise_and(slika_hsv, slika_hsv, mask=maska)
filter_slika_bgr = cv2.cvtColor(filter_slika_hsv, cv2.COLOR_HSV2BGR)

cv2.imshow('maska', maska)
cv2.imshow('filter', filter_slika_bgr)
cv2.waitKey(9000)
cv2.destroyAllWindows()

# %% zadaatak 7

slika7 = cv2.imread('kamera2.jpg')
grayscale = cv2.cvtColor(slika7, cv2.COLOR_BGR2GRAY)
slika_zamucena = cv2.GaussianBlur(grayscale, (5, 5), 0)

# ovo mi nije jasno
# nova_slika = cv2.fastNlMeansDenoising(slika_zamucena, None, 10, 10, 7)
# sobel_x_abs = cv2.convertScaleAbs(nova_slika)

sobel_x = cv2.Sobel(slika_zamucena, cv2.CV_64F, 1, 0, ksize=5)
sobel_x_abs = cv2.convertScaleAbs(sobel_x)


cv2.imshow("original", slika7)
cv2.imshow("sobel_x", sobel_x_abs)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% zzadaatak 8
slika8 = cv2.imread('kamera2.jpg')

canny = cv2.Canny(slika8, 100, 200)

cv2.imshow("original", slika8)
cv2.imshow("canny", canny)
cv2.waitKey(5000)
cv2.destroyAllWindows()

# %% novi batch zadataka - zadatak 1
slika9 = cv2.imread('kamera3.jpeg', cv2.IMREAD_GRAYSCALE)

canny = cv2.Canny(slika9, 100, 200)

lines = cv2.HoughLinesP(canny, 1, np.pi / 180, 100,
                        minLineLength=100, maxLineGap=10)

prazna = np.zeros_like(slika9)

mask = np.zeros_like(slika9)
height, width = slika9.shape

x1 = width / 2
y1 = height / 2
x2 = 0
y2 = height
x3 = width
y3 = height

rectangle_points = np.array(
    [[(x1, y1), (x2, y2), (x3, y3)]], dtype=np.int32)

mask = cv2.fillPoly(mask, rectangle_points, 255)
masked_image = cv2.bitwise_and(slika9, mask)

for line in lines:
    cv2.line(prazna, (line[0][0], line[0][1]),
             (line[0][2], line[0][3]), (255, 0, 0), 1)
    cv2.line(masked_image, (line[0][0], line[0][1]),
             (line[0][2], line[0][3]), (255, 0, 0), 1)

cv2.imshow("original", masked_image)
cv2.imshow("linija", prazna)
cv2.waitKey()
cv2.destroyAllWindows()


# %%
