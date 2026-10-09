"""Genera las figuras del README a partir de los resultados obtenidos en el notebook."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SALIDA = Path(__file__).resolve().parent.parent / "images"
SALIDA.mkdir(exist_ok=True)
VERDE, ROJO, GRIS = "#2e7d32", "#c62828", "#9e9e9e"

# --- Problema 1: análisis de sentimientos ---
sentimientos = [
    ("Funciona perfectamente y resolvió rápido", "Positivo", "Positivo", 0.6690),
    ("Excelente atención, técnico amable", "Positivo", "Positivo", 0.9633),
    ("Horas intentando ingresar, sin solución", "Negativo", "Negativo", 0.9429),
    ("Pésimo servicio, tercera vez que reporto", "Negativo", "Negativo", 0.9505),
    ("La actualización fue instalada", "Neutral", "Neutral", 0.8746),
    ("Solucionaron, aunque tardó demasiado", "Ambiguo/Mixto", "Neutral", 0.6283),
]
fig, ax = plt.subplots(figsize=(10, 4.5))
etiquetas = [f"{t}\n(esperado: {e} | predicho: {p})" for t, e, p, _ in sentimientos]
colores = [VERDE if e == p else GRIS for _, e, p, _ in sentimientos]
ax.barh(etiquetas, [c for *_, c in sentimientos], color=colores)
for i, (*_, c) in enumerate(sentimientos):
    ax.text(c + 0.01, i, f"{c:.2f}", va="center")
ax.set_xlim(0, 1.08)
ax.invert_yaxis()
ax.set_xlabel("Confianza")
ax.set_title("Problema 1 · Sentimiento con RoBERTuito (5/6 aciertos; gris = caso mixto)")
fig.tight_layout()
fig.savefig(SALIDA / "p1_sentimientos.png", dpi=150)
plt.close(fig)

# --- Problema 2: clasificación zero-shot de tickets ---
tickets = [
    ("App se cierra al generar reporte", "Software", "Software", 0.9541),
    ("No reconoce segundo monitor HDMI", "Hardware", "Hardware", 0.5657),
    ("VPN conecta pero sin acceso al servidor", "Redes", "Redes", 0.6564),
    ("Correo pidiendo credenciales (phishing)", "Seguridad", "Acceso y cuentas", 0.5102),
    ("Contraseña expirada", "Acceso y cuentas", "Seguridad", 0.5531),
    ("Portátil mojado no enciende", "Hardware", "Hardware", 0.7930),
    ("Error 500 en facturación", "Software", "Software", 0.5666),
    ("Login desde ubicación desconocida", "Seguridad", "Acceso y cuentas", 0.5427),
    ("Internet de la oficina se cae", "Redes", "Redes", 0.4575),
]
fig, ax = plt.subplots(figsize=(10, 5.5))
etiquetas = [f"{t}\n(esperado: {e} | predicho: {p})" for t, e, p, _ in tickets]
colores = [VERDE if e == p else ROJO for _, e, p, _ in tickets]
ax.barh(etiquetas, [c for *_, c in tickets], color=colores)
for i, (*_, c) in enumerate(tickets):
    ax.text(c + 0.01, i, f"{c:.2f}", va="center")
ax.set_xlim(0, 1.08)
ax.invert_yaxis()
ax.set_xlabel("Confianza de la categoría top-1")
ax.set_title("Problema 2 · Zero-shot con mDeBERTa-v3 (6/9 aciertos; rojo = error)")
fig.tight_layout()
fig.savefig(SALIDA / "p2_tickets.png", dpi=150)
plt.close(fig)

# --- Problema 3: resumen extractivo (similitud coseno oración × criterio) ---
criterios = ["problema", "impacto", "causa", "acciones", "solución"]
scores = np.array([
    [0.2827, 0.5225, 0.1969, 0.1927, 0.3656],
    [0.3173, 0.3250, 0.1901, 0.2098, 0.2517],
    [0.4953, 0.5541, 0.4137, 0.3895, 0.4450],
    [0.3043, 0.3591, 0.2611, 0.1839, 0.2965],
    [0.2737, 0.2996, 0.2209, 0.1588, 0.2789],
    [0.2339, 0.3211, 0.2296, 0.1637, 0.2521],
    [0.3902, 0.5121, 0.5450, 0.2768, 0.3506],
    [0.2005, 0.3391, 0.0731, 0.3061, 0.5894],
    [0.1883, 0.2546, 0.1180, 0.1985, 0.2177],
    [0.1745, 0.1826, 0.1378, 0.2051, 0.2913],
    [0.1694, 0.2471, -0.0195, 0.1736, 0.4902],
    [0.3174, 0.3107, 0.2266, 0.3425, 0.2629],
])
seleccionadas = {0, 2, 6, 7, 10}
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(scores, cmap="viridis", aspect="auto")
ax.set_xticks(range(len(criterios)), criterios)
ax.set_yticks(
    range(len(scores)),
    [f"Oración {i}{'  ★' if i in seleccionadas else ''}" for i in range(len(scores))],
)
for i in range(scores.shape[0]):
    for j in range(scores.shape[1]):
        ax.text(j, i, f"{scores[i, j]:.2f}", ha="center", va="center",
                color="white" if scores[i, j] < 0.4 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="Similitud coseno")
ax.set_title("Problema 3 · Resumen extractivo con MiniLM\n(★ = oración seleccionada; 59.6% de reducción)")
fig.tight_layout()
fig.savefig(SALIDA / "p3_resumen_extractivo.png", dpi=150)
plt.close(fig)

# --- Resumen global de desempeño ---
problemas = ["P1 Sentimiento", "P2 Tickets", "P3 Resumen\nabstractivo", "P3 Resumen\nextractivo", "P4 Traducción"]
aciertos = [5 / 6, 6 / 9, 0.0, 1.0, 5 / 6]
textos = ["5/6", "6/9", "Alucina\n(4 intentos)", "5/5 elementos\nconservados", "5/6"]
colores = [VERDE, "#f9a825", ROJO, VERDE, VERDE]
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.bar(problemas, aciertos, color=colores)
for i, (a, t) in enumerate(zip(aciertos, textos)):
    ax.text(i, a + 0.03, t, ha="center")
ax.set_ylim(0, 1.25)
ax.set_ylabel("Proporción de casos correctos")
ax.set_title("Resumen de desempeño de los modelos preentrenados")
fig.tight_layout()
fig.savefig(SALIDA / "resumen_desempeno.png", dpi=150)
plt.close(fig)

print("Figuras generadas en", SALIDA)
