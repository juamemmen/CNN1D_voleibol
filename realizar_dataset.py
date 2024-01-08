import pandas as pd
from numpy import genfromtxt
import matplotlib.pyplot as plt


# Definir las constantes
ventana = 50

min_golpes=44

# Archivos de entrada
archivos = ["preprocesado_dedos.csv", "preprocesado_saque.csv", "preprocesado_antebrazos.csv", "preprocesado_ataque.csv"]
tipos_de_golpe = ["dedos", "saque", "antebrazos", "ataque"]  # Agrega los tipos de golpes correspondientes

# Inicializar un DataFrame vacío
df_final = pd.DataFrame()

# Leer y unir datos de cada archivo
for i, archivo in enumerate(archivos):
    datos_IMU = genfromtxt(archivo, delimiter=',')

    # Limitar la lectura al total de muestras deseadas
    t = datos_IMU[0]
    accel_x = datos_IMU[1]
    accel_y = datos_IMU[2]
    accel_z = datos_IMU[3]
    gyros_x = datos_IMU[4]
    gyros_y = datos_IMU[5]
    gyros_z = datos_IMU[6]
    

    # Crear un DataFrame para cada archivo
    df_temp = pd.DataFrame({
        'tipo_golpe': tipos_de_golpe[i],  # Agregar el tipo de golpe correspondiente
        't': t,
        'accel_x': accel_x,
        'accel_y': accel_y,
        'accel_z': accel_z,
        'gyros_x': gyros_x,
        'gyros_y': gyros_y,
        'gyros_z': gyros_z
        
    })

    # Unir el DataFrame actual al DataFrame final
    df_final = pd.concat([df_final, df_temp], ignore_index=True)
    

# Guardar el DataFrame final en un nuevo archivo CSV
df_final.to_csv('datos_combinados_nuevo.csv', index=False, header=False)
