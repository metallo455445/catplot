#py ./catenaria/catenaria.py ./catenaria/catenaTerraPulita.png 20 ./catenaria/coord.txt
#^py     ^percorso file .py          ^percorso file di rif     ^numero di bordi da mostare   ^percorso del file su cui salvare le coordinate
#                                                                                                                   se non esiste, lo crea
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys

# Ottieni il percorso del file dall'argomento
if len(sys.argv) > 1:
    file_path = sys.argv[1]
    #contour_index = int(sys.argv[2]) if len(sys.argv) > 2 else 4   #<--Non uso più questo variabile, la rimuovo
    n_top_countours = int(sys.argv[2]) if len(sys.argv) > 2 else 5  #aggionato il nomero dell'argv da 3 a 2
    coord_path = sys.argv[3]                                        #aggiunto, in effetti è comodo passarlo da terminale
else:
    file = input("Inserisci il nome del file: ")
    file_path = './' + file
    #contour_index = 4
    n_top_countours = 5
    coord_path = './coord.txt'

open(coord_path, 'w').close()   #formatta il file delle coord prima di riscrivere
#percorso file coordinate punti della parabola da fittare

# Usa file_path ovunque
img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
assert img is not None, f"file {file_path} could not be read, check file path/integrity"
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

titles = ['Original Image [0]', 'Global Thresholding (v = 127)[1]',
            'Adaptive Mean Thresholding[2]', 'Adaptive Gaussian Thresholding[3]']
images = [img, th1, th2, th3]

plt.figure("Correzione impurezze")
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])

#edge detection partendo dall'immagine pura         <---
plt.figure("edge detection pura")
edges = cv.Canny(img, 100, 200)

edgesPURE = edges

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

#edge detection post pulizia dell'adattiva gaussiana
plt.figure("edge detection from gaussian")
edges = cv.Canny(th3, 100, 200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# edges è già in scala di grigi
ret, thresh = cv.threshold(th1, 127, 255, 0)          #<--- ho messo qui il edges per evitare sporcizia
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

# Stampa informazioni sui contorni
print(f"Numero totale di contorni trovati: {len(contours)}")

# Carica l'immagine originale a colori
img_color = cv.imread(file_path)

if img_color is not None and len(contours) > 0:
    contours_sorted = sorted(
        enumerate(contours),
        key=lambda x: cv.contourArea(x[1]),
        reverse=True
    )
    
    # Disegna TUTTI i contorni
    img_all_contours = img_color.copy()
    cv.drawContours(img_all_contours, contours, -1, (0,255,0), 2)
    
    plt.figure("Tutti i contorni")
    plt.imshow(cv.cvtColor(img_all_contours, cv.COLOR_BGR2RGB))
    plt.title(f'Tutti i {len(contours)} contorni')
    
    # Disegna i primi n contorni più grandi (escludendo eventualmente il bordo dell'immagine)
    img_top_contours = img_color.copy()
    # Salta il primo se è troppo grande (potrebbe essere il bordo)
    start_idx = 1 if len(contours_sorted) > 1 and cv.contourArea(contours_sorted[0][1]) > img.shape[0] * img.shape[1] * 0.9 else 0    
    
    for i in range(start_idx, min(start_idx + n_top_countours, len(contours_sorted))):                    ## start_idx + 5 sostituito da me coon start_idx + n_top_countours
        cv.drawContours(img_top_contours, [contours_sorted[i][1]], 0, (0,255,0), 3)        
    
    top_ids = [idx for idx, _ in contours_sorted[start_idx:start_idx + n_top_countours]]              ## start_idx + 5 sostituito da me coon start_idx + n_top_countours
    print(f"\nID dei top {n_top_countours} contorni:", top_ids)

    plt.figure(f"Top {n_top_countours} contorni")
    plt.imshow(cv.cvtColor(img_top_contours, cv.COLOR_BGR2RGB))
    plt.title(f'Top {n_top_countours} contorni più grandi')

    # Inizializza un dizionario globale per raccogliere i punti PRIMA di getCoord
    coordinate_grezze = {}

    def getCoord(indiceContornoTarget):
        print(f"Coords di contorno {indiceContornoTarget}")
        if len(contours) > indiceContornoTarget:       
            cnt = contours[indiceContornoTarget]
            img_single = img_color.copy()
            
            # Altezza dell'immagine per l'inversione
            height = img.shape[0] 

            for point in contours[indiceContornoTarget]:
                x_raw = int(point[0][0])
                y_raw = int(point[0][1])
                
                # INVERSIONE ASSE Y
                x = x_raw
                y = height - y_raw  # Ora 0 è il fondo dell'immagine
                
                p_plot = (x_raw, y_raw)
                p_calc = (x, y)
                
                # INVECE DI SALVARE SU FILE, RAGGRUPPA LE Y PER OGNI X
                if x not in coordinate_grezze:
                    coordinate_grezze[x] = [] # Crea una nuova lista se la X non esiste
                coordinate_grezze[x].append(y) # Aggiunge la Y alla colonna X
                
                # Disegno
                cv.circle(img_single, p_plot, 2, (255, 0, 0), -1)

            cv.drawContours(img_single, [cnt], 0, (0,0,255), 3)
            plt.figure(f"Contorno {indiceContornoTarget}")
            plt.imshow(cv.cvtColor(img_single, cv.COLOR_BGR2RGB))
            plt.title(f'Contorno {indiceContornoTarget} (Visualizzazione Standard)')

    def selezioneContorni(select):
        i = 0
        for id in top_ids:
            if id not in select:
                getCoord(id)
#####
    #INCOLLA QUI I getCoord()
    selezione = np.array([613, 287, 612, 487, 505, 523,
                          419, 250, 405, 207, 379, 372,
                          132, 531, 213, 468, 656, 271,
                          696, 667, 408, 602, 112, 216,
                          388, 417, 628, 642])
    selezioneContorni(selezione)

    #stampa tutti i contorni selezionati (scartando gli errori)
    img_contorni_finali = img_color.copy()
    
    # Cicliamo su tutti i contorni trovati all'inizio
    for id_corrente in top_ids:
        # Se l'ID NON è nella lista degli scarti, allora è un pezzo buono della catena!
        if id_corrente not in selezione:
            if id_corrente < len(contours): 
                # (0, 0, 255) è il colore rosso, 3 è lo spessore
                cv.drawContours(img_contorni_finali, [contours[id_corrente]], 0, (0, 0, 255), 3)

    # Creazione della finestra Matplotlib dedicata
    plt.figure("Catenaria Finale Selezionata")
    plt.imshow(cv.cvtColor(img_contorni_finali, cv.COLOR_BGR2RGB))
    
    # Calcoliamo quanti contorni sono sopravvissuti per il titolo
    contorni_buoni = len(top_ids) - len(selezione)
    plt.title(f"I {contorni_buoni} contorni salvati per il Fit")
    plt.axis('off')

    # CALCOLO DELLA LINEA CENTRALE E SALVATAGGIO DEFINITIVO
    print("\nCalcolo della linea centrale in corso...")
    
    # Riscrive il file 'w' (sovrascrive quello vuoto creato all'inizio)
    with open(coord_path, "w") as f:
        # Ordina le X dalla più piccola alla più grande (utile per i fit successivi!)
        for x_corrente in sorted(coordinate_grezze.keys()):
            lista_y = coordinate_grezze[x_corrente]
            
            # Calcola la media delle Y per questa X
            y_media = sum(lista_y) / len(lista_y)
            
            # Scrive sul file la X e la Y media (con 2 cifre decimali)
            f.write(f"{x_corrente} {y_media:.2f}\n")
            
    print(f"File {coord_path} generato con successo! Punti mediati salvati.")
else:
    print("Impossibile caricare l'immagine a colori o nessun contorno trovato")

plt.show()
