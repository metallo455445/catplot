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

#Nel caso in cui la cartella temp non esista, la creo
Path(bContours).parent.mkdir(parents=True, exist_ok=True)

if len(sys.argv) > 1:
    file_path = sys.argv[1]
    #contour_index = int(sys.argv[2]) if len(sys.argv) > 2 else 4   #<--Non uso più questo variabile, la rimuovo
    n_top_countours = int(sys.argv[2]) if len(sys.argv) > 2 else 5  #aggionato il nomero dell'argv da 3 a 2
    #coord_path = sys.argv[3]                                        #aggiunto, in effetti è comodo passarlo da terminale
else:
    file = input("Insert file path: ")
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

fig = plt.figure("Noise Correction", figsize=(10, 6))

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
    print(f"Selected file: {label}")

radio.on_clicked(cleaning_selector)

plt.tight_layout()

plt.subplots_adjust(right=0.95)

print("Select the desired image from the Radio Buttons.")
print("Once the choice is made, CLOSE the window to proceed with edge detection.")
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

#omentaneamente qui, poi riordino
def click_reset(event):
    # Svuota il file dei contorni scartati
    open(bContours, 'w').close()
    print(f"File {bContours} reset! All of the contours will be reconsidered.")
    plt.close()


def click_continue(event):
    plt.close()

# Stampa informazioni sui contorni
print(f"Total number of contours found: {len(contours)}")

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
    
    plt.figure("All Contours")
    plt.imshow(cv.cvtColor(img_all_contours, cv.COLOR_BGR2RGB))
    plt.title(f"All {len(contours)} contours")
    
    # Disegna i primi n contorni più grandi (escludendo eventualmente il bordo dell'immagine)
    img_top_contours = img_color.copy()
    # Salta il primo se è troppo grande (potrebbe essere il bordo)
    start_idx = 1 if len(contours_sorted) > 1 and cv.contourArea(contours_sorted[0][1]) > img.shape[0] * img.shape[1] * 0.9 else 0    
    
    for i in range(start_idx, min(start_idx + n_top_countours, len(contours_sorted))):                    ## start_idx + 5 sostituito da me coon start_idx + n_top_countours
        cv.drawContours(img_top_contours, [contours_sorted[i][1]], 0, (0,255,0), 3)        
    
    top_ids = [idx for idx, _ in contours_sorted[start_idx:start_idx + n_top_countours]]              ## start_idx + 5 sostituito da me coon start_idx + n_top_countours
    print(f"\nID of the top {n_top_countours} contours:", top_ids)

    top_cont = plt.figure(f"Top {n_top_countours} contours")

    asse_principale = plt.axes([0.1, 0.25, 0.8, 0.7])
    
    asse_principale.imshow(cv.cvtColor(img_top_contours, cv.COLOR_BGR2RGB))
    asse_principale.set_title(f'Top {n_top_countours} contours')
    asse_principale.axis('off')

    top_cont.text(0.5, 0.20, "Add new contours or start over?", ha='center', fontsize=12)

    asse_btn1 = plt.axes([0.25, 0.05, 0.2, 0.1])
    asse_btn2 = plt.axes([0.55, 0.05, 0.2, 0.1])

    btn1 = Button(asse_btn1, 'RESET')
    btn2 = Button(asse_btn2, 'CONTINUE')

    btn1.on_clicked(click_reset)
    btn2.on_clicked(click_continue)

    plt.show()

    # Inizializza un dizionario globale per raccogliere i punti PRIMA di getCoord
    coordinate_grezze = {}

    # 1. NUOVA LISTA GLOBALE per proteggere i bottoni dal Garbage Collector
    active_buttons = []

    # 2. LEGGIAMO I CONTORNI SCARTATI PRIMA DI getCoord() (senza usare np.loadtxt per evitare crash!)
    if Path(bContours).exists():
        with open(bContours, 'r') as f:
            # Crea una normale lista Python, a prova di errore
            selezione = [int(line.strip()) for line in f if line.strip().isdigit()]
    else:
        selezione = []

    def getCoord(indiceContornoTarget):
        print(f"Coords of contour {indiceContornoTarget}")
        if len(contours) > indiceContornoTarget:       
            cnt = contours[indiceContornoTarget]
            img_single = img_color.copy()
            
            height = img.shape[0] 

            # Creiamo una lista temporanea per i punti di questo contorno
            punti_temporanei = []

            for point in contours[indiceContornoTarget]:
                x_raw = int(point[0][0])
                y_raw = int(point[0][1])
                
                x = x_raw
                y = height - y_raw
                
                punti_temporanei.append((x, y))
                
                cv.circle(img_single, (x_raw, y_raw), 2, (255, 0, 0), -1)

            cv.drawContours(img_single, [cnt], 0, (0,0,255), 3)
            fig_scelta = plt.figure(f"Contour {indiceContornoTarget}")

            ax_img = plt.axes([0.1, 0.25, 0.8, 0.7])
            ax_img.imshow(cv.cvtColor(img_single, cv.COLOR_BGR2RGB))
            ax_img.set_title(f'Contour {indiceContornoTarget}')
            ax_img.axis('off')

            fig_scelta.text(0.5, 0.18, "Is the image correct? Press OK or NO.", ha='center', fontsize=12)

            ax_btn1 = plt.axes([0.25, 0.05, 0.2, 0.1])
            ax_btn2 = plt.axes([0.55, 0.05, 0.2, 0.1])

            # Creiamo i bottoni locali
            btn_ok = Button(ax_btn1, 'OK')
            btn_no = Button(ax_btn2, 'NO')

            # INSERIAMOLI NELLA LISTA GLOBALE per non farli distruggere dal sistema
            active_buttons.clear()
            active_buttons.extend([btn_ok, btn_no])

            def locale_click_ok(event):
                # Trasferiamo i punti in coordinate_grezze SOLO se premi OK
                for px, py in punti_temporanei:
                    if px not in coordinate_grezze:
                        coordinate_grezze[px] = []
                    coordinate_grezze[px].append(py)
                plt.close(fig_scelta)

            def locale_click_no(event):
                # Scriviamo l'ID nel file di testo
                with open(bContours, 'a') as file:
                    file.write(f"{indiceContornoTarget}\n")
                print(f"Discarded: added number {indiceContornoTarget} to file {bContours}!")
                
                # 3. FIX VISIVO: AGGIORNIAMO LA LISTA 'selezione' IN TEMPO REALE!
                # Altrimenti il programma lo disegnerà comunque nella finestra finale
                if indiceContornoTarget not in selezione:
                    selezione.append(indiceContornoTarget)
                    
                plt.close(fig_scelta)

            # Colleghiamo le funzioni ai bottoni
            btn_ok.on_clicked(locale_click_ok)
            btn_no.on_clicked(locale_click_no)

            plt.show()

    def selezioneContorni(select):
        for id in top_ids:
            if id not in select:
                getCoord(id)

    # Avviamo il processo
    selezioneContorni(selezione)

    # Stampa tutti i contorni selezionati (scartando gli errori)
    img_contorni_finali = img_color.copy()
    
    for id_corrente in top_ids:
        # Adesso, siccome abbiamo aggiornato la lista 'selezione' in tempo reale, i contorni scartati non verranno disegnati!
        if id_corrente not in selezione:
            if id_corrente < len(contours): 
                cv.drawContours(img_contorni_finali, [contours[id_corrente]], 0, (0, 0, 255), 3)

    plt.figure("Final Selected Contours")
    plt.imshow(cv.cvtColor(img_contorni_finali, cv.COLOR_BGR2RGB))
    
    contorni_buoni = len(top_ids) - len(selezione)
    plt.title(f"I {contorni_buoni} contours saved for the Fit")
    plt.axis('off')

    print("\nCalcolo della linea centrale in corso...")
    
    with open(coord_path, "w") as f:
        for x_corrente in sorted(coordinate_grezze.keys()):
            lista_y = coordinate_grezze[x_corrente]
            y_media = sum(lista_y) / len(lista_y)
            f.write(f"{x_corrente} {y_media:.2f}\n")
            
    with open(error_path, "w") as f:
        for x_corrente in sorted(coordinate_grezze.keys()):
            lista_y = coordinate_grezze[x_corrente]
            error = (max(lista_y) - min(lista_y)) / np.sqrt(12) 
            f.write(f"{error:.2f}\n")

    print(f"File {coord_path} generated successfully! Median points saved.")
else:
    print("Unable to load the color image or no contours found")

plt.show()
