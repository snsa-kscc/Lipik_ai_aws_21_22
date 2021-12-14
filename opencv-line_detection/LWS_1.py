'''
Zadatak 1 - implementacija osnovnog LDWS

14.12.2021.
'''

# ovdje definirajte dodatne biblioteke ako su vam potrebne
import numpy as np
import cv2
import math


# TODO: napisite funkciju za detekciju rubova; funkcija vraca binarnu sliku s detektiranim rubovima
def detectEdges(image):

    canny_image = cv2.Canny(image, 100, 120)
    # dummy rezultat  - obrisite
    # canny_image = 1

    return canny_image


# TODO: napisite funkciju za filtriranje po boji u HLS prosotru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
def filterByColor(image):
    # TODO: pretvorite sliku iz BGR u HLS
    img_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # TODO: definirajte granice za bijelu boju te kreirajte masku pomocu funkcije cv2.inRange
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([180, 255, 255])
    white_mask = cv2.inRange(img_hls, lower, upper)
    # TODO: definirajte granice za zutu boju te kreirajte masku pomocu funkcije cv2.inRange
    lower_yellow = np.uint8([20, 80, 80])
    upper_yellow = np.uint8([32, 255, 255])
    yellow_mask = cv2.inRange(img_hls, lower_yellow, upper_yellow)
    # TODO: kombinirajte obje maske pomocu odgovarajuce logicke operacije (bitwise)
    mask = cv2.bitwise_or(yellow_mask, white_mask)
    # TODO: filtirajte sliku pomocu dobivene maske koristei odgovarajucu logicku operaciju (bitwise)
    result = cv2.bitwise_and(image, image, mask=mask)
    # dummy rezultat  - obrisite
    # result = yellow_mask = mask = 1

    return result, yellow_mask, white_mask, mask


# TODO: napisite funkciju za pronalazenje pravaca lijeve i desne kolnice oznake
# ulaz je binarna slika, a izlaz dvije liste koje sadrze pravce koji pripadaju lijevoj odnosnoj desnoj kolnickoj oznaci
def findLines(img):

    # TODO: koristite cv2.HoughLinesP() kako biste dobili linije na slici
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, 15,
                            minLineLength=100, maxLineGap=10)
    # lines = 1

    # od svih linija treba pronaci one koje predstavljaju lijevu odnosno desnu uzduznu kolnicku oznaku
    linesLeft = []
    linesRight = []

    # TODO: pokusajte razumjeti iduci kod; mozete li odgonetnuti cemu sluzi pojedini dio?
    try:
        for line in lines:

            x1, y1, x2, y2 = line[0]
            if abs(x2-x1) <= 1.0:   # ako je linija okomita
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


# TODO: dovrsite funkciju koja oznacava sa zelenom povrsinom voznu traku (podrucje unutar pravaca) te ispisuje upozorenje na originalni ulazni frame
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
            # TODO: dodajte overlay pomocu funkcije cv2.addWeighted()
            cv2.addWeighted(overlay, 0.35, frameToDraw, 0.65, 0, frameToDraw)

    if linesLeft:
        if linesLeft[0][2] == 1:

            # TODO: koristite funkcije cv2.putText kako biste na ekranu crvenim slovima ispisali upozorenje
            print("Upozorenje")

    if linesRight:
        if linesRight[0][2] == 1:
            print("Upozorenje")

    return frameToDraw


def punInfoImg(img, text, loc):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (0, 0, 255)
    lineType = 2

    cv2.putText(img, text, loc, font,
                fontScale, fontColor, lineType)


pathResults = 'results/'
pathVideos = 'videos/'
videoName = 'video2.mp4'

# TODO: otvorite video pomocu cv2.VideoCapture
cap = cv2.VideoCapture(videoName)

# TODO: spremite sirinu i visinu videa u varijable width i height
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# TODO: ovdje otvorite prozore za prikaz video signala i ostale rezultate (neka bude tipa cv2.WINDOW_NORMAL)
cv2.namedWindow('Input', cv2.WINDOW_NORMAL)
cv2.namedWindow('RoI', cv2.WINDOW_NORMAL)

# ovdje definirajte sve ostale varijable po potrebi koje su vam potrebne za razvoj rjesenja
k = 0
RoIymin = 460
RoIymax = 620

while True:
    e1 = cv2.getTickCount()
    # TODO: ucitaj frame pomocu metode read, povecaj k za jedan ako je uspjesno ucitan frame
    ret, frame = cap.read()
    if ret == False:
        print("Video end")
        break
    else:
        k += 1

    # TODO: kreiraj regiju od interesa (RoI) izdvajanjem dijela numpy polja koje predstavlja frame
    frameRoI = frame[RoIymin:RoIymax, 0:width, :]
    # TODO: pozovite funkciju za filtriranje po boji
    result, yellow_mask, white_mask, mask = filterByColor(frameRoI)
    # TODO: pozovite funkciju za detekciju rubova na filtriranoj slici kako bi ste smanjili kolicinu piksela koji se dalje procesiraju
    RoI_filtered = detectEdges(result)
    # TODO: pozovite funkciju za pronalazak pravaca lijeve i desne linije na slici s rubovima
    linesLeft, linesRight = findLines(RoI_filtered)
    # TODO: pozovite funkciju za prikaz vozne trake
    RoILines = drawLane(linesLeft, linesRight, frame)
    # TODO: prikazi frame pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite

    cv2.imshow("Input", frame)
    # cv2.imshow("RoI", frameRoI)
    cv2.imshow("combined mask", result)
    cv2.imshow("canny", RoI_filtered)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    # TODO: ovdje ispisite vrijeme procesiranja jednog okvira
    e2 = cv2.getTickCount()
    time = (e2 - e1) / cv2.getTickFrequency()
    print("Vrijeme obrade u fps: ", 1.0/time)


# TODO: ovdje unistite sve prozore i oslobodite objekt koji je kreiran pomocu cv2.VideoCapture
cap.release()
cv2.destroyAllWindows()
