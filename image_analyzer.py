import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
from pathlib import Path
from matplotlib.widgets import Button, RadioButtons

# percorso del file dall'argomento
script_dir = Path(__file__).parent.absolute()

coord_path = str(script_dir) + '/temp/coord.txt'
error_path = str(script_dir) + '/temp/errors.txt'
bContours  = str(script_dir) + '/temp/bad_cont.txt'

if len(sys.argv) > 1:
    file_path = sys.argv[1]
    #contour_index = int(sys.argv[2]) if len(sys.argv) > 2 else 4   #<--Non uso più questo variabile, la rimuovo
    n_top_countours = int(sys.argv[2]) if len(sys.argv) > 2 else 5  #aggionato il nomero dell'argv da 3 a 2
    #coord_path = sys.argv[3]                                        #aggiunto, in effetti è comodo passarlo da terminale
else:
    file = input("Inserisci il nome del file: ")
    file_path = './' + file
    #contour_index = 4
    n_top_countours = 5
    coord_path = './coord.txt'

open(coord_path, 'w').close()   #formatta il file delle coord prima di riscrivere

# Usa i filtri
img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
assert img is not None, f"file {file_path} could not be read, check file path/integrity"
img = cv.medianBlur(img,5)

ret,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY)
th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

selected_image = img  # Default to the original image

fig = plt.figure("Correzione impurezze", figsize=(10, 6))

for i in range(4):
    row = i // 2
    col = (i % 2) * 2  
    
    ax = plt.subplot2grid((2, 5), (row, col), colspan=2)
    ax.imshow(images[i], 'gray')
    ax.set_title(titles[i])
    ax.set_xticks([])
    ax.set_yticks([])

ax_radio = plt.subplot2grid((2, 5), (0, 4), rowspan=2, facecolor='0.95')
radio = RadioButtons(ax_radio, ['Original Image', 'Global', 'Mean', 'Gaussian'])

def cleaning_selector(label):
    global selected_image
    if label == 'Original Image':
        selected_image = img
    elif label == 'Global':
        selected_image = th1
    elif label == 'Mean':
        selected_image = th2
    elif label == 'Gaussian':
        selected_image = th3
    print(f"Filtro selezionato: {label}")

radio.on_clicked(cleaning_selector)

plt.tight_layout()

plt.subplots_adjust(right=0.95)

print("Seleziona l'immagine desiderata dai Radio Buttons.")
print("Una volta fatta la scelta, CHIUDI la finestra per proseguire con l'edge detection.")
plt.show()

#edge detection partendo dall'immagine pura         <---
plt.figure("edge detection")
edges = cv.Canny(selected_image, 100, 200)

plt.subplot(121),plt.imshow(selected_image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

# edges è già in scala di grigi
ret, thresh = cv.threshold(edges, 127, 255, 0)          #<--- ho messo qui il edges per evitare sporcizia
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

    def click_ok(event):
        plt.close()

    def click_no(numero, nome_file):
        with open(nome_file, 'a') as file:
            file.write(f"{numero}\n")
        print(f"Aggiunto il numero {numero} al file {nome_file}!")
        plt.close()


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
            fig_scelta = plt.figure(f"Contorno {indiceContornoTarget}")

            #asse principale per l'immagine
            ax_img = plt.axes([0.1, 0.25, 0.8, 0.7])

            #stampa immagine nell'asse appena creato
            ax_img.imshow(cv.cvtColor(img_single, cv.COLOR_BGR2RGB))
            ax_img.set_title(f'Contorno {indiceContornoTarget} (Visualizzazione Standard)')
            ax_img.axis('off')

            # Testo informativo opzionale tra l'immagine e i bottoni
            fig_scelta.text(0.5, 0.18, "L'immagine è corretta? Premi OK o NO.", ha='center', fontsize=12)

            ax_btn1 = plt.axes([0.25, 0.05, 0.2, 0.1])
            ax_btn2 = plt.axes([0.55, 0.05, 0.2, 0.1])

            btn1 = Button(ax_btn1, 'OK')
            btn2 = Button(ax_btn2, 'NO')

            btn1.on_clicked(click_ok)
            btn2.on_clicked(lambda event: click_no(indiceContornoTarget, bContours))

            plt.show()


    def selezioneContorni(select):
        for id in top_ids:
            if id not in select:
                getCoord(id)

    #preleva i contorni da scartare dal file di testo (se esiste) e li passa alla funzione selezioneContorni
    selezione = np.loadtxt(bContours, dtype=int) if Path(bContours).exists() else []

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
            
    with open(error_path, "w") as f:
        # Ordina le X dalla più piccola alla più grande (utile per i fit successivi!)
        for x_corrente in sorted(coordinate_grezze.keys()):
            lista_y = coordinate_grezze[x_corrente]
            
            # errore distribuzione uniforme
            error = (max(lista_y) - min(lista_y)) / np.sqrt(12) 
            
            # Scrive sul file la X e la Y media (con 2 cifre decimali)
            f.write(f"{error:.2f}\n")

    print(f"File {coord_path} generato con successo! Punti mediati salvati.")
else:
    print("Impossibile caricare l'immagine a colori o nessun contorno trovato")

plt.show()
