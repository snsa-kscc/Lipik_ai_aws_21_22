'''
Zadatak 1 - implementacija osnovnog LDWS 
 
14.12.2021.
'''

# ovdje definirajte dodatne datoteke ako su vam potrebne
import numpy as np
import cv2
import math


# TODO: Napisite funkciju za detekciju rubova; funkcija vraca binarnu sliku s detektiranim rubovima
def detectEdges(image):

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blurred_img, 100, 120)

    return canny_image


# TODO: Napisite funkciju za filtriranje po boji u HLS prostoru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
def filterByColor(image):
    # TODO: Pretvorite sliku iz BGR u HLS
    imgHLS = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    # TODO: Definirajte granice za bijelu boju te kreirajte masku pomocu funkcije cv2.inRange
    lower = np.uint8([0, 200,   0])
    upper = np.uint8([180, 255, 255])
    white_mask = cv2.inRange(imgHLS, lower, upper)

    # TODO: Definirajte granice za zutu boju te kreirajte masku pomocu funkcije cv2.inRange
    lower = np.uint8([20,   0, 100])
    upper = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(imgHLS, lower, upper)

    # TODO: Kombinirajte obje maske pomocu odgovarajuce logicke operacije (bitwise)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # TODO: Filtirajte sliku pomocu dobivene maske koristei odgovarajucu logicku operaciju (bitwise)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result, yellow_mask, white_mask, mask


# TODO: Napisite funkciju za pronalazenje pravaca lijeve i desne kolnice oznake
# ulaz je binarna slika, a izlaz dvije liste koje sadrze pravce koji pripadaju lijevoj odnosnoj desnoj kolnickoj oznaci
def findLines(img):

    # TODO: Koristite cv2.HoughLinesP() kako biste dobili linije na slici
    lines = cv2.HoughLinesP(img, 1, np.pi/180, 15,
                            minLineLength=10, maxLineGap=200)

    # TODO: Pronadite od svih linija one koje predstavljaju lijevu odnosno desnu uzduznu kolnicku oznaku
    linesLeft = []
    linesRight = []

    try:
        for line in lines:

            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) <= 1.0:
                b = np.inf
                a = np.inf
                x_val = x1
                lineAngle = 90.0
            else:
                a = (y2-y1)/(x2-x1)
                b = y1 - a*x1
                x_val = (img.shape[0] - b)/a
                lineAngle = math.atan2((y2-y1), (x2-x1)) * 180/np.pi

            if x_val > 150.0 and x_val < 1200.0:

                # lijeva i desna linija
                if lineAngle > 10.0 and lineAngle <= 90.0:
                    if x_val > 450.0 and x_val < 800.0:
                        linesRight.append([a, b, 1, x_val])
                    else:
                        linesRight.append([a, b, 0, x_val])
                elif lineAngle < -10.0 and lineAngle >= -90.0:
                    if x_val > 450.0 and x_val < 800.0:
                        linesLeft.append([a, b, 1, x_val])
                    else:
                        linesLeft.append([a, b, 0, x_val])
    except:
        linesRight = []
        linesLeft = []

    return linesRight, linesLeft

# TODO: Napisite funkciju koja oznacava sa zelenom povrsinom voznu traku (podrucje unutar pravaca) te ispisuje upozorenje na originalni ulazni frame


def drawLane(linesLeft, linesRight, frameToDraw):

    ymin = 0
    ymax = frameToDraw.shape[0]

    if linesLeft and linesRight:

        if linesLeft[0][1] != np.inf and linesLeft[0][1] != np.inf:

            x1_1 = int((ymin - linesLeft[0][1]) / linesLeft[0][0])
            x1_2 = int((ymax - linesLeft[0][1]) / linesLeft[0][0])
        else:
            x1_1 = linesLeft[0][3]
            x1_2 = linesLeft[0][3]

        if linesRight[0][1] != np.inf and linesRight[0][1] != np.inf:

            x2_1 = int((ymin - linesRight[0][1]) / linesRight[0][0])
            x2_2 = int((ymax - linesRight[0][1]) / linesRight[0][0])
        else:
            x2_1 = linesRight[0][3]
            x2_2 = linesRight[0][3]

        if linesLeft[0][2] == 0 and linesRight[0][2] == 0:
            contours = np.array([[x1_1, ymin+RoIymin], [x2_1, ymin+RoIymin],
                                [x2_2, ymax+RoIymin], [x1_2, ymax+RoIymin]])
            overlay = frameToDraw.copy()

            cv2.fillPoly(overlay, [contours], color=(0, 255, 100))
            cv2.addWeighted(overlay, 0.35, frameToDraw,
                            1 - 0.35, 0, frameToDraw)

    if linesLeft:
        if linesLeft[0][2] == 1:
            cv2.putText(frameToDraw, "Upozorenje!", (int(width/2)-140, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_4)

    if linesRight:
        if linesRight[0][2] == 1:
            cv2.putText(frameToDraw, "Upozorenje!", (int(width/2)-140, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_4)

    return frameToDraw


def putInfoImg(img, text, loc):

    cv2.putText(img,
                text,
                (loc[0], loc[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255),
                2,
                cv2.LINE_4)


pathResults = 'results/'
pathVideos = 'videos/'
videoName = 'video5.mp4'

# TODO: Otvorite video pomocu cv2.VideoCapture
cap = cv2.VideoCapture(videoName)

# TODO: Spremite sirinu i visinu video okvira u varijable width i height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# TODO: Otvorite prozore za prikaz video signala i ostale rezultate (neka bude tipa cv2.WINDOW_NORMAL)
cv2.namedWindow('Input image', cv2.WINDOW_NORMAL)
cv2.namedWindow('RoI', cv2.WINDOW_NORMAL)
cv2.namedWindow('RoI_Color_Filtered', cv2.WINDOW_NORMAL)
cv2.namedWindow('RoI_Color_Filtered_Edges', cv2.WINDOW_NORMAL)

# ovdje definirajte sve ostale varijable po potrebi koje su vam potrebne za razvoj rjesenja
k = 0
RoIymin = 460
RoIymax = 620
frameToSave = 35
time = 1

while True:
    e1 = cv2.getTickCount()

    # TODO: Ucitaj video okvir pomocu metode read, povecaj k za jedan ako je uspjesno ucitan
    ret, frame = cap.read()
    if ret == False:
        print("Video end!")
        break
    else:
        k = k + 1

    # TODO: Kreiraj regiju od interesa (RoI) izdvajanjem dijela numpy polja koje predstavlja video okvir
    frameRoI = frame[RoIymin:RoIymin+(RoIymax-RoIymin), 0:width, :]

    # TODO: Pozovite funkciju za filtriranje po boji RoI-a
    RoIFiltered, yellow_mask, white_mask, mask = filterByColor(frameRoI)

    # TODO: Pozovite funkciju za detekciju rubova na filtriranoj slici kako bi ste smanjili kolicinu piksela koji se dalje procesiraju
    RoIFilteredEdges = detectEdges(RoIFiltered)

    # TODO: Pozovite funkciju za pronalazak pravaca lijeve i desne linije na slici s rubovima
    linesRight, linesLeft = findLines(RoIFilteredEdges)

    # TODO: Pozovite funkciju za prikaz vozne trake na ulaznom video okviru
    RoILines = drawLane(linesLeft, linesRight, frame)

    putInfoImg(frame, "frame: " + str(k), ((50, 50)))
    putInfoImg(frame, "FPS: " + str(int(1/time)), ((50, 100)))

    # TODO: Prikazi video okvir pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite
    cv2.imshow('Input image', frame)
    cv2.imshow('RoI', frameRoI)
    cv2.imshow('RoI_Color_Filtered', RoIFiltered)
    cv2.imshow('RoI_Color_Filtered_Edges', RoIFilteredEdges)

    if k == frameToSave:
        cv2.imwrite("frame_%d.jpg" % k, RoILines)     # save frame as JPEG file

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # TODO: Ovdje ispisite vrijeme procesiranja jednog video okvira
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("Vrijeme obrade u fps: ", 1.0/time)


# TODO: Ovdje unistite sve prozore i oslobodite objekt koji je kreiran pomocu cv2.VideoCapture
cap.release()
cv2.destroyAllWindows()
