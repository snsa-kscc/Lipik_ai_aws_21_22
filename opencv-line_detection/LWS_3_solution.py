'''
LDWS koji koristi transformaciju perspektive i zakrivljene linije

14.12.2021.
'''
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt


def plotArea(image, pts):

    for i in range(0,4):
        cv2.circle(image, (pts[i,0],pts[i,1]), radius=5, color=(255, 0, 0), thickness=-1)
    
    cv2.line(image, (pts[0,0], pts[0,1]), (pts[1,0],pts[1,1]), (0, 255, 0), thickness=2)
    cv2.line(image, (pts[1,0], pts[1,1]), (pts[2,0],pts[2,1]), (0, 255, 0), thickness=2)
    cv2.line(image, (pts[2,0], pts[2,1]), (pts[3,0],pts[3,1]), (0, 255, 0), thickness=2)
    cv2.line(image, (pts[3,0], pts[3,1]), (pts[0,0],pts[0,1]), (0, 255, 0), thickness=2)

    
def putInfoImg(img, text, loc):

    cv2.putText(img, 
                text, 
                (loc[0], loc[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 255), 
                2, 
                cv2.LINE_4)


# TODO: napisite funkciju za filtriranje po boji u HLS prostoru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
def filterByColor(image):

    imgHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([180, 255, 255])
    white_mask = cv2.inRange(imgHLS, lower, upper)

    lower = np.uint8([ 20,   0, 100])
    upper = np.uint8([ 30, 255, 255])
    yellow_mask = cv2.inRange(imgHLS, lower, upper)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    result = cv2.bitwise_and(image, image, mask = mask)
    
    return result, yellow_mask, white_mask, mask


# TODO: Napisite funkciju koja detektira dva maksimuma u sumi binarne slike po "vertikali" koji sluze kao pocetak metode pomicnih prozora
# koristite samo donju polovicu binarne slike kako biste tocnije detektirali pocetak krivulje
def getTwoPeaks(binary_img):

    column_sums = binary_img[int(binary_img.shape[0]/2):,:].sum(axis=0)

    x1 = np.argmax(column_sums)

    x1_1 = x1 - 150
    x1_2 = x1 + 150

    if x1_1 < 0:
        x1_1 = 0
    if x1_2 > len(column_sums):
        x1_2 = len(column_sums)

    column_sums[x1_1:x1_2] = 0

    x2 = np.argmax(column_sums)
    
    if x1 > x2:
        x_left = x2
        x_right = x1
    else:
        x_left = x1
        x_right = x2

    return x_left, x_right


# TODO: Napisite funkciju primjenjuje metodu pomicnog prozora; funkcija vraca koeficijent lijevi i desne krivulje drugog reda
# Npr. funkcija lijeve krivulje je oblika: x = leftLine[0] * (y**2) + leftLine[1] * y + leftLine[2]
def slidingWindow(binary_img, color_img, x_left, x_right):
    height = binary_img.shape[0]
    windowSize_X_2 = 90
    windowSize_Y_2 = 30

    noWindows = int(height/(windowSize_Y_2*2))
    
    #prozor krece od dolje
    y_pos = height - windowSize_Y_2
    x_pos = x_left

    # ako nema piksela unutar prozora koriste se stare vrijednosti pozicije prozora
    x_last = x_pos
    y_last = y_pos

    leftPtsX = np.empty((0,1), int)
    leftPtsY = np.empty((0,1), int)
    for i in range(0, noWindows):

        cv2.rectangle(color_img,(x_pos-windowSize_X_2,y_pos-windowSize_Y_2),(x_pos+windowSize_X_2,y_pos+windowSize_Y_2),(0,255,0),3)
       
        y, x = np.where(binary_img[y_pos-windowSize_Y_2 : y_pos+windowSize_Y_2, x_pos-windowSize_X_2 : x_pos+windowSize_X_2]==255)
        if x.size > 0 and y.size > 0:
            leftPtsX = np.append(leftPtsX, np.reshape(x+x_pos-windowSize_X_2,(len(x),1)), axis=0)
            leftPtsY = np.append(leftPtsY, np.reshape(y+y_pos-windowSize_Y_2,(len(y),1)), axis=0)
            x_pos = int(np.mean(x)) + x_pos-windowSize_X_2
        else:
            x_pos = x_last
        
        y_pos = y_last - windowSize_Y_2*2

        x_last = x_pos
        y_last = y_pos

    if leftPtsX.shape[0] > 3:
        leftLine = np.polyfit( np.reshape(leftPtsY,(len(leftPtsY),)), np.reshape(leftPtsX,(len(leftPtsX),)), 2)
    else:
        leftLine = np.zeros((3,1))

    
    # start je dolje
    y_pos = height - windowSize_Y_2
    x_pos = x_right

    # ako nema piksela koriste se stare vrijednosti pozicije prozora
    x_last = x_pos
    y_last = y_pos

    rightPtsX = np.empty((0,1), int)
    rightPtsY = np.empty((0,1), int)
    for i in range(0, noWindows):

        cv2.rectangle(color_img,(x_pos-windowSize_X_2,y_pos-windowSize_Y_2),(x_pos+windowSize_X_2,y_pos+windowSize_Y_2),(0,255,0),3)
       
        y, x = np.where(binary_img[y_pos-windowSize_Y_2 : y_pos+windowSize_Y_2, x_pos-windowSize_X_2 : x_pos+windowSize_X_2]==255)
        if x.size > 0 and y.size > 0:
            rightPtsX = np.append(rightPtsX, np.reshape(x+x_pos-windowSize_X_2,(len(x),1)), axis=0)
            rightPtsY = np.append(rightPtsY, np.reshape(y+y_pos-windowSize_Y_2,(len(y),1)), axis=0)
            x_pos = int(np.mean(x)) + x_pos-windowSize_X_2
               
        else:
            x_pos = x_last
        
        y_pos = y_last - windowSize_Y_2*2  

        x_last = x_pos
        y_last = y_pos
    
    if rightPtsX.shape[0] > 3:
        rightLine = np.polyfit( np.reshape(rightPtsY,(len(rightPtsY),)), np.reshape(rightPtsX,(len(rightPtsX),)), 2)
    else:
        rightLine = np.zeros((3,1))
    
    return leftLine, rightLine


# TODO: Napisite funkciju koja crta krivulju drugog reda na transformiranu sliku
# Najprije izracunajte tocke koje su na krivulji uniformnim uzorkovanjem y osi, a zatim dobivene tocke spojite pomocu cv2.line()
def plotPolyLine(warped_img, leftLine, rightLine):

    ys = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0])
    left_xs = leftLine[0] * (ys**2) + leftLine[1] * ys + leftLine[2]
    right_xs = rightLine[0] * (ys**2) + rightLine[1] * ys + rightLine[2]
    xls, xrs, ys = left_xs.astype(np.uint32), right_xs.astype(np.uint32), ys.astype(np.uint32)

    try:
        for xl, xr, y in zip(xls, xrs, ys):
            cv2.line(warped_img, (xl - 4, y), (xl + 4, y), (255, 255, 0), 2)
            cv2.line(warped_img, (xr - 4, y), (xr + 4, y), (0, 0, 255), 2)
    except:
        print("some error")   

    left_xs = np.expand_dims(left_xs, axis=1)
    right_xs = np.expand_dims(right_xs, axis=1)
    ys = np.expand_dims(ys, axis=1)
    
    return left_xs, right_xs, ys


# TODO: Napisite funkciju koja oznacava voznu traku na ulaznom video okviru
# tocke (x,y) transformirajte u originalni video okvir, a zatim povrsinu ispunite s cv2.fillPoly()
def showLane(original_img, left_xs, right_xs, ys, M_inv):

    src_left = np.append(left_xs, ys, axis = 1)
    src_right = np.append(right_xs, ys, axis = 1)

    dst_left = cv2.perspectiveTransform(np.array([src_left]), M_inv)
    dst_right = cv2.perspectiveTransform(np.array([src_right]), M_inv)

    dst_left = dst_left[0,:,:]
    dst_right = dst_right[0,:,:]
    pts = np.append(dst_left, np.flip(dst_right, axis=0), axis=0)
    pts = pts.reshape((-1,1,2)).astype(np.int32)

    overlay = original_img.copy()
    cv2.fillPoly(overlay, [pts], color=(0, 255, 100))
    cv2.addWeighted(overlay, 0.35, original_img, 1 - 0.35, 0, original_img)

    return 0




pathResults = 'results/'
pathVideos = 'videos/'
videoName  = 'video1.mp4'

cv2.namedWindow('Frame',cv2.WINDOW_NORMAL)
cv2.namedWindow('Warped frame',cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(videoName)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

src = np.array([
    [190,700],
    [580, 450],
    [730, 450],
    [1160, 700]],
    dtype = np.float32)

dst = np.array([
    [380, 720],
    [380, 10],
    [950, 10],
    [950, 720]],
    dtype = np.float32)


src1 = np.array([
    [205, 720],
    [1110, 720],
    [703, 460],
    [580, 460]],
    dtype = np.float32)

dst1 = np.array([
    [320, 720],
    [960, 720],
    [960, 0],
    [320, 0]],
    dtype = np.float32)

M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

frameToSave = 100
k = 0
time = 1
while(True):
    e1 = cv2.getTickCount()

    # ucitaj frame
    ret, frame = cap.read()
    if ret == False:
        print("Video end!")
        break
    else:
        k = k + 1
    
    frame_cpy = frame.copy()

    # za debuggiranje 
    #plotArea(frame, src1)    

    # TODO: napravi transformaciju perspektive
    warped_img = cv2.warpPerspective(frame[:-50,:], M, (width,height-50), flags=cv2.INTER_LINEAR)

    # TODO: Pozovite funkciju za filtriranje po boji nad ulaznim okvirom
    frameFiltered, yellow_mask, white_mask, mask = filterByColor(warped_img)

    # TODO: Pozovite funkciju koja detektira dva maksimuma u sumi binarne slike po "vertikali" koji sluze kao pocetak metode pomicnih prozora
    x_left, x_right = getTwoPeaks(mask)
    
    # TODO: Pozovite funkciju koja metodom pomicnih prozora racuna krivulju drugog reda koja odgovara lijevoj odnosno desnoj kolnickoj oznaci
    leftLine, rightLine = slidingWindow(mask, frameFiltered, x_left, x_right)

    # TODO: Pozovite funkciju koja za zakrivljene linije racuna tocke (x,y) i prikazuje ih krivulju
    left_xs, right_xs, ys = plotPolyLine(frameFiltered, leftLine, rightLine)

    # TODO: Pozovite funkciju koja transformira dobivene tocke u originalni video okvir i spaja ih ravnim linijama 
    showLane(frame, left_xs, right_xs, ys, M_inv)

    # za debuggiranje
    #if k==frameToSave:
    #    plt.figure(1)
    #    x_array = range(0,mask.shape[1])
    #    column_sums = mask[int(mask.shape[0]/2):,:].sum(axis=0)
    #    plt.plot(x_array, column_sums)
    #    plt.xlabel("x pozicija")
    #    plt.ylabel("suma")
    #    plt.show()
   
    # TODO: Prikazite okvir pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite
    putInfoImg(frame, "frame: " + str(k), ((50,50)))
    putInfoImg(frame, "FPS: " + str(int(1/time)), ((50,100)))

    cv2.imshow('Frame',frame)
    cv2.imshow('Warped frame',warped_img)
    cv2.imshow('Binary', frameFiltered)

    if k == frameToSave:
        cv2.imwrite("frame_%d.jpg" % k, frame)     

    key =  cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break

    if key == ord('p'):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            if key2 == ord('p'):
                break
            if key2 == ord('q'):
                break
    
    e2 = cv2.getTickCount()
    time = (e2 - e1)/ cv2.getTickFrequency()
    

cap.release()
cv2.destroyAllWindows()
