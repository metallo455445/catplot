#py ./catenaria/FITcatenaria.py ./catenaria/coord.txt
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import chi2
import matplotlib as mpl
import sys

# --- GESTIONE INPUT ---
if len(sys.argv) > 1:
    coord_path = sys.argv[1]
    bg_path = sys.argv[2]
    img1 = sys.argv[3]

else:
    # Fallback per test o errore
    print("Nessun file specificato, uso percorso default o esco.")
    # coord_path = "./catenaria/coord.txt" # Decommenta se vuoi un default
    sys.exit(1)
    img1 = False

#convertitore da stringa a bool
if img1 == 'True' or img1 == 'true':
    img1 = True
elif img1 == 'False' or img1 == 'false':
    img1 = False

if img1:
    mpl.use("pgf")

    plt.rcParams.update({
        "font.family": "serif",     
        "text.usetex": True,
        "pgf.rcfonts": False,
    })

data = np.loadtxt(coord_path, delimiter=" ")

image = plt.imread('photos/exp1.jpg')

# --- CALCOLO ERRORI ---
# Sensibilità dello strumento (pixel). 
# Un valore di 1.0 assume un'incertezza di +/- 1 pixel su ogni punto.
errore_x = 4.
errore_y = 4. 

# np.full crea un array lungo quanto i dati, riempito con il valore dell'errore
devStdX = np.full(len(data[:, 0]), errore_x)
devStdY = np.full(len(data[:, 1]), errore_y)

sx = devStdX
sy = devStdY

print(f"Errore X impostato a: {errore_x} pixel")
print(f"Errore Y impostato a: {errore_y} pixel")

# --- DEFINIZIONE FUNZIONI CATENARIA ---

def catenaria(x, a, b, c):
    """
    a = parametro di forma (tensione / densità)
    b = traslazione orizzontale (coordinata x del vertice)
    c = traslazione verticale (offset y)
    Formula: a * cosh((x - b) / a) + c
    """
    return a * np.cosh((x - b) / a) + c

def derivata_catenaria(x, a, b):
    """
    Derivata prima della catenaria rispetto a x.
    d/dx [ a * cosh((x - b) / a) + c ] = sinh((x - b) / a)
    """
    return np.sinh((x - b) / a)

# Dati
x_data = data[:, 0]
y_data = data[:, 1]
sx = devStdX  
sy = devStdY

# --- STIMA INIZIALE PARAMETRI (Guess) ---
p0_a = 100.0                # Valore arbitrario di partenza per la curvatura
p0_b = np.mean(x_data)      # Il minimo è circa al centro dei dati X
p0_c = np.min(y_data) - 100 # Il minimo verticale è circa il minimo dei dati Y
p0_guess = [p0_a, p0_b, p0_c]

# --- PROCESSO DI FIT ITERATIVO ---

print("Inizio fit catenaria...")

# Fit iniziale (solo errori su Y)
try:
    popt, pcov = curve_fit(catenaria, x_data, y_data, p0=p0_guess, sigma=sy, absolute_sigma=True, maxfev=5000)
except RuntimeError:
    print("Il fit iniziale non è riuscito a convergere. Controlla i dati o i parametri iniziali.")
    sys.exit(1)

# Raffinamento con Incertezza Efficace
for i in range(5):
    a, b, c = popt
    
    # Calcolo della derivata (Sinh)
    df_dx = derivata_catenaria(x_data, a, b)
    
    # Incertezza efficace
    sigma_eff = np.sqrt(sy**2 + (df_dx * sx)**2)
    
    # Nuovo fit pesato
    try:
        popt, pcov = curve_fit(catenaria, x_data, y_data, p0=popt, sigma=sigma_eff, absolute_sigma=True, maxfev=5000)
    except:
        pass # Se fallisce un'iterazione, mantiene i parametri precedenti

a_fin, b_fin, c_fin = popt
perr = np.sqrt(np.diag(pcov)) 

print(f"Parametri ottimizzati:\n a (shape) = {a_fin:.4f}\n b (x0)    = {b_fin:.4f}\n c (y0)    = {c_fin:.4f}")
print(f"pcvo: {pcov}")

# --- Calcolo Residui e Chi2 ---
y_model = catenaria(data[:, 0], a_fin, b_fin, c_fin)

# Sigma efficace finale per il chi2
df_dx_final = derivata_catenaria(data[:, 0], a_fin, b_fin)
sigma_eff_final = np.sqrt(devStdY**2 + (df_dx_final * devStdX)**2)

# Residui normalizzati (pull)
residui = (data[:, 1] - y_model) / sigma_eff_final

# Chi2
chi2_calc = np.sum(residui**2)
ndof = len(data[:, 0]) - len(popt)
chi2_ridotto = chi2_calc / ndof
p_value = chi2.sf(chi2_calc, ndof)

print(f"chi2: {chi2_calc:.2f}")
print(f"Gradi di libertà (ndof): {ndof}")
print(f"chi2 ridotto: {chi2_ridotto:.2f}")

# --- Grafico con Sfondo Immagine ---
fig1 = plt.figure(1)
#plt.figure("Verifica Visiva Fit", figsize=(10, 10))

#Carica l'immagine come sfondo
img = plt.imread(bg_path) ##!!!!
plt.imshow(img, cmap='gray') # Mostra l'immagine

# Traccia i dati grezzi
# Poiché imshow ha già l'origine in alto a sinistra, e i dati sono invertiti, 
# dobbiamo "re-invertirli" per metterli nel posto giusto.
# L'inversione 'height - y' è stata fatta in catenaria.py.
# Qui, per visualizzarli sull'immagine, non facciamo nulla.
# Ma assicurati di usare 'p_plot' (le coordinate originali) se vuoi disegnarli direttamente.
# Visto che hai salvato 'p_calc' (Y invertito) nel file, qui dobbiamo re-invertirli.
height = img.shape[0]
#plt.errorbar(data[:,0], height - data[:,1], yerr=devStdY, xerr=devStdX, fmt='+', alpha=0.5, label='Dati', color='blue')


# Generazione e tracciamento della linea del fit (sull'asse Y invertito)
x_plot = np.linspace(min(data[:,0]), max(data[:,0]), 1000)
# Tracciamo height - catenaria(x_plot, ...) per sovrapporla correttamente all'immagine
plt.plot(x_plot, height - catenaria(x_plot, a_fin, b_fin, c_fin), color='red', label='Fit Catenaria')

# Impostazioni finali del grafico
plt.title(f"Fit Catenaria su Immagine")
plt.legend()
plt.axis('off') # Nascondi gli assi per una visualizzazione pulita
plt.tight_layout()
#plt.show()

fig2, (ax1, ax2) = plt.subplots(2, 1, sharex=True, height_ratios=[3, 1], figsize=(8, 8))

# Grafico Best Fit
ax1.errorbar(data[:,0], data[:,1], yerr=devStdY, xerr=devStdX, fmt='+', alpha=0.5, label='Dati')

# Generazione linea fit
x_plot = np.linspace(min(data[:,0]), max(data[:,0]), 1000)
ax1.plot(x_plot, catenaria(x_plot, a_fin, b_fin, c_fin), color='red', label='Fit Catenaria')
ax1.plot([], [], ' ', label=rf'$\chi^2$ridotto: {chi2_ridotto:.2f}')
ax1.plot([], [], ' ', label=rf'p-value: {p_value:.2f}')
ax1.legend()
ax1.grid(ls='dashed')
ax1.set_ylabel("Y")

# Grafico Residui
ax2.errorbar(data[:,0], residui, yerr=1, fmt='o', markersize=3, alpha=0.6, color="#b561fe")
ax2.axhline(0, color='black', linestyle='dashed')
ax2.set_ylabel("Residui norm. ($\sigma$)")
ax2.set_xlabel("X")
ax2.grid(ls='dashed')

plt.tight_layout()

if img1:
    fig2.savefig('grafico_fit.pgf')
    fig1.savefig('fit_back.pdf')
else:
    plt.show()
