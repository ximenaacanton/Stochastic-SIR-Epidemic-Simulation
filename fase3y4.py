import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 1000  # Num de personas
D = 300   # Tamaño ciudad
r_infeccion = 5  # Radio de infección
tasa_recuperacion = 0.2 
num_iteraciones = 10 
movimiento_sd = 5 

#distancia euclidiana
def dist_euc(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Inicializar población
def inicializar_poblacion(N, D, distribucion="uniforme", cluster_centro=None):
    if distribucion == "uniforme":
        x = np.random.uniform(0, D, N)
        y = np.random.uniform(0, D, N)
    elif distribucion == "circular":
        theta = np.random.uniform(0, 2 * np.pi, N)
        r = np.sqrt(np.random.uniform(0, (D / 2) ** 2, N))
        x = r * np.cos(theta)
        y = r * np.sin(theta)
    elif distribucion == "cluster" and cluster_centro is not None:
        x_centro, y_centro = cluster_centro
        x = np.random.normal(x_centro, D / 20, N)
        y = np.random.normal(y_centro, D / 20, N)
        # Limites
        x = np.clip(x, 0, D)
        y = np.clip(y, 0, D)
    else:
        raise ValueError("Distribución no reconocida o falta de centro para cluster")
    
    #estados iniciales
    estados = np.array(["Infectado"] * 10 + ["Susceptible"] * (N - 10))
    np.random.shuffle(estados)
    
    return pd.DataFrame({
        "ID": np.arange(1, N + 1),
        "x": x,
        "y": y,
        "estado": estados,
        "iteracion": 0
    })

# Actualizar estados y posiciones
def actualizar_poblacion(poblacion, r_infeccion, tasa_recuperacion, D, distribucion="cuadrado"):
    N = len(poblacion)
    dx = np.random.normal(0, movimiento_sd, N)
    dy = np.random.normal(0, movimiento_sd, N)
    poblacion["x"] += dx
    poblacion["y"] += dy
    
    #posiciones límites
    if distribucion == "cuadrado":
        poblacion["x"] = np.clip(poblacion["x"], 0, D)
        poblacion["y"] = np.clip(poblacion["y"], 0, D)
    elif distribucion == "circular":
        distancias = np.sqrt(poblacion["x"] ** 2 + poblacion["y"] ** 2)
        fuera = distancias > D / 2
        poblacion.loc[fuera, "x"] *= (D / 2) / distancias[fuera]
        poblacion.loc[fuera, "y"] *= (D / 2) / distancias[fuera]
    
    # Propagación infección
    infectados = poblacion[poblacion["estado"] == "Infectado"]
    susceptibles = poblacion[poblacion["estado"] == "Susceptible"]
    
    for i, infectado in infectados.iterrows():
        for j, susceptible in susceptibles.iterrows():
            if dist_euc(infectado["x"], infectado["y"], susceptible["x"], susceptible["y"]) < r_infeccion:
                poblacion.at[j, "estado"] = "Infectado"
    
    # Actualizar estado
    poblacion.loc[infectados.index, "estado"] = np.where(
        np.random.rand(len(infectados)) < tasa_recuperacion,
        "Recuperado",
        "Infectado"
    )
    return poblacion

# Visualización
def animar_poblacion(poblacion_inicial, num_iteraciones, D, r_infeccion, tasa_recuperacion, distribucion):
    poblacion = poblacion_inicial.copy()
    historial = [poblacion.copy()]
    
    for _ in range(num_iteraciones):
        poblacion = actualizar_poblacion(poblacion, r_infeccion, tasa_recuperacion, D, distribucion)
        historial.append(poblacion.copy())
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    scatter = ax.scatter(poblacion["x"], poblacion["y"], c=poblacion["estado"].map({"Susceptible": "blue", "Infectado": "red", "Recuperado": "green"}), alpha=0.6)
    ax.set_xlim(0, D)
    ax.set_ylim(0, D)
    ax.set_title("Iteración: 0")
    
    ax.set_xlabel("Posición X (metros)")
    ax.set_ylabel("Posición Y (metros)")
    
    handles, labels = scatter.legend_elements()
    ax.legend(handles, ["Susceptible", "Infectado", "Recuperado"], title = "Estados")
    
    def actualizar(frame):
        iter_poblacion = historial[frame]
        scatter.set_offsets(iter_poblacion[["x", "y"]].values)
        scatter.set_array(iter_poblacion["estado"].map({"Susceptible": 0, "Infectado": 1, "Recuperado": 2}).values)
        ax.set_title(f"Iteración: {frame}")
        return scatter,
    
    anim = FuncAnimation(fig, actualizar, frames=num_iteraciones, interval=500, blit=False)
    plt.show()
    return anim

# Problema 1: Ciudad cuadrada con distribución uniforme
poblacion = inicializar_poblacion(N, D, "uniforme")
animar_poblacion(poblacion, num_iteraciones, D, r_infeccion, tasa_recuperacion, "cuadrado")

#Problema 2: 
poblacion = inicializar_poblacion(N, D, "circular")
animar_poblacion(poblacion, num_iteraciones, D, r_infeccion, tasa_recuperacion, "circular")

#Problema 3:
poblacion = inicializar_poblacion(N, D,"cluster", cluster_centro=(D/2,D/2))
animar_poblacion(poblacion, num_iteraciones, D, r_infeccion, tasa_recuperacion, "cuadrado")

#Problema 4:
poblacion = inicializar_poblacion(N, D, "cluster", cluster_centro=(0,0))
animar_poblacion(poblacion, num_iteraciones, D, r_infeccion, tasa_recuperacion, "circular")