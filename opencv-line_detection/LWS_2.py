'''
 Zadatak 2 - implementacija LDWS koji koristi transformaciju perspektive
 
14.12.2021.
'''


# ovdje definirajte dodatne datoteke ako su vam potrebne
import numpy as np
import cv2
import matplotlib.pyplot as plt


# TODO: napisite funkciju koja oznacava 4 tocke na ulaznoj slici i spaja ih pravcima - za provjeru 4 tocke perspektivne transformacije
def plotArea(image, pts):

    return


# TODO: napisite funkciju za filtriranje po boji u HLS prostoru
# ulaz je slika u boji, funkcija vraca binarnu sliku te maske za bijelu, zutu boju i ukupnu masku
def filterByColor(image):

    # dummy rezultat  - obrisite
    result = yellow_mask = white_mask = mask = 1

    return result, yellow_mask, white_mask, mask


# TODO: napisite funkcija koja detektira dva maksimuma u sumi binarne slike po "vertikali"
def getTwoPeaks(binary_img):

    # dummy rezultat  - obrisite
    x_left = x_right = 1

    return x_left, x_right


# TODO: prikazite voznu traku u ulaznoj slici; ako vozilo prelazi u drugu traku tada iskljucite prikaz i ispisite upozorenje
def showLane(original_img, x_left, x_right, y1, y2, M_inv):

    return 
    



pathResults = 'results/'
pathVideos = 'videos/'
videoName  = 'video2.mp4'

# TODO: Otvorite video pomocu cv2.VideoCapture


# TODO: Spremite sirinu i visinu video okvira u varijable width i height


# TODO: Otvorite prozore za prikaz video signala i ostale rezultate (neka bude tipa cv2.WINDOW_NORMAL)


# TODO: Definirajte 4 tocke u ulaznoj slici u numpy polju src, 4 tocke u izlaznoj slici u dst numpy polju
# Izracunajte matricu perspektivne transformacije (M) i njen inverz (M^-1)


# TODO: ucitajte sliku i pozovite funkciju koja crta 4 tocke na ulaznoj slici i spaja ih pravcima - kako biste bili sigurni u tocke koje se koriste u transfromaciji
# Trebate najprije pohraniti jedan reprezentativni okvir na disk iz danog video signala

k = 0

while(True):


    # TODO: Ucitaj video okvir (frame) pomocu metode read, povecaj k za jedan ako je uspjesno ucitan


    # TODO: Pozovite funkciju za filtriranje po boji nad ulaznim okvirom


    # TODO: Transformirajte filtriranu binarnu sliku


    # TODO: Pozovite funkciju koja pronalazi dva maksimuma u "vertikalnoj sumi" transformirane binarne slike


    # TODO: Pozovite funkciju koja oznacava voznu traku u originalnom video okviru; u slucaju prelaska u drugu ispisuje upozorenje 


    # TODO: Izracunajte vrijeme obrade u fps


    # TODO: Prikazite vrijeme obrade i redni broj okvira u gornjem lijevom cosku ulaznog video okvira


    # TODO: Prikazite okvir pomocu cv2.imshow(); i sve ostale medjurezultate kada ih napravite

    key =  cv2.waitKey(1) & 0xFF 
    if key == ord('q'):
        break



# TODO: Unistite sve prozore i oslobodite objekt koji je kreiran pomocu cv2.VideoCapture