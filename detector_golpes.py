# -*- coding: utf-8 -*-

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

umbral_golpe_accel =2 # umbrales que se han tenido en cuenta para la detección
umbral_golpe_vel = 5
hit_counter = 0
ventana = 50  # muestras que se consideran en cada golpe
last_hit_t = ventana // 2 + 1
find_hit = True

archivo = "saque_marina.csv"  # archivo de la prueba
datos_IMU = genfromtxt(archivo, delimiter=',')

t = datos_IMU[0]  # cada característica es una fila
accel_x = datos_IMU[1]
accel_y = datos_IMU[2]
accel_z = datos_IMU[3]
gyros_x = datos_IMU[4]
gyros_y = datos_IMU[5]
gyros_z = datos_IMU[6]

hit_detection = np.zeros(len(t))  # vamos a guardar el tiempo de detección
hit_samples = []  # lista para almacenar muestras de golpes

for i in range(len(t) - ventana // 2):  # tiempo de espera entre golpes
    if t[i] <= t[last_hit_t]:
        find_hit = False
    else:
        find_hit = True

    if (abs(gyros_x[i]) > umbral_golpe_vel or  # Comprobación de umbrales
        abs(gyros_y[i]) > umbral_golpe_vel or
        abs(gyros_z[i]) > umbral_golpe_vel) and \
       (abs(accel_x[i]) > umbral_golpe_accel or
        # abs(accel_y[i]) > umbral_golpe_accel or
        abs(accel_z[i]) > umbral_golpe_accel) and find_hit:
        hit_detection[i] = 1
        hit_counter += 1
        find_hit = False

        last_hit_t = i + ventana // 2  # no se pueden detectar dos golpes seguidos
        # Guardar las muestras del golpe en hit_samples
        hit_samples.append({
            't': t[i - ventana // 2:i + ventana // 2],
            'accel_x': accel_x[i - ventana // 2:i + ventana // 2],
            'accel_y': accel_y[i - ventana // 2:i + ventana // 2],
            'accel_z': accel_z[i - ventana // 2:i + ventana // 2],
            'gyros_x': gyros_x[i - ventana // 2:i + ventana // 2],
            'gyros_y': gyros_y[i - ventana // 2:i + ventana // 2],
            'gyros_z': gyros_z[i - ventana // 2:i + ventana // 2]
        })
        

# Guardar las muestras de golpes en variables separadas
all_t = []
all_accel_x = []
all_accel_y = []
all_accel_z = []
all_gyros_x = []
all_gyros_y = []
all_gyros_z = []

for idx, hit_sample in enumerate(hit_samples):
    all_t.extend(hit_sample['t'])
    all_accel_x.extend(hit_sample['accel_x'])
    all_accel_y.extend(hit_sample['accel_y'])
    all_accel_z.extend(hit_sample['accel_z'])
    all_gyros_x.extend(hit_sample['gyros_x'])
    all_gyros_y.extend(hit_sample['gyros_y'])
    all_gyros_z.extend(hit_sample['gyros_z'])

# Crear un archivo CSV con todas las muestras de golpes
if hit_counter > 0:
    all_hit_data = np.array([all_t, all_accel_x, all_accel_y, all_accel_z, all_gyros_x, all_gyros_y, all_gyros_z])

    np.savetxt("preprocesado_saque_pruebas.csv", all_hit_data, delimiter=',', fmt='%s')
    hit_sample = hit_samples[0]  # Tomar el primer golpe

    plt.figure()
    plt.subplot(211)
    plt.plot(hit_sample['t'], hit_sample['accel_x'], 'b')
    plt.plot(hit_sample['t'], hit_sample['accel_y'], 'r')
    plt.plot(hit_sample['t'], hit_sample['accel_z'], 'y')
    plt.legend(['x', 'y', 'z'])
    plt.grid()
    plt.ylabel('Accel (Gs) - Golpe Detectado')

    plt.subplot(212)
    plt.plot(hit_sample['t'], hit_sample['gyros_x'], 'b')
    plt.plot(hit_sample['t'], hit_sample['gyros_y'], 'r')
    plt.plot(hit_sample['t'], hit_sample['gyros_z'], 'y')
    plt.legend(['x', 'y', 'z'])
    plt.grid()
    plt.xlabel('Time (s)')
    plt.ylabel('Gyros (rad/s) - Golpe Detectado')

    plt.show()



# Representar las muestras de los primeros 5 golpes detectados
if hit_counter >= 5:
    plt.figure(figsize=(15, 10))

    for i in range(5):
        hit_sample = hit_samples[i]  # Tomar el i-ésimo golpe

        # Crear una nueva figura para cada golpe
        plt.figure()

        # Gráficas para el i-ésimo golpe
        plt.subplot(211)
        plt.plot(hit_sample['t'], hit_sample['accel_x'], 'b')
        plt.plot(hit_sample['t'], hit_sample['accel_y'], 'r')
        plt.plot(hit_sample['t'], hit_sample['accel_z'], 'y')
        plt.legend(['x', 'y', 'z'])
        plt.grid()
        plt.ylabel(f'Accel (Gs) - Golpe {i + 1}')
        plt.title(f'Golpe {i + 1}')

        plt.subplot(212)
        plt.plot(hit_sample['t'], hit_sample['gyros_x'], 'b')
        plt.plot(hit_sample['t'], hit_sample['gyros_y'], 'r')
        plt.plot(hit_sample['t'], hit_sample['gyros_z'], 'y')
        plt.legend(['x', 'y', 'z'])
        plt.grid()
        plt.xlabel('Time (s)')
        plt.ylabel(f'Gyros (rad/s) - Golpe {i + 1}')

    plt.show()

    

# Resto del código para la representación de los golpes detectados y la impresión del número de golpes
plt.figure()
plt.subplot(311)
plt.plot(t, accel_x, 'b')
plt.plot(t, accel_y, 'r')
plt.plot(t, accel_z, 'y')
plt.legend(['x', 'y', 'z'])
plt.grid()
plt.ylabel('Accel (Gs)')
plt.title('Experimento completo')

plt.subplot(312)
plt.plot(t, gyros_x, 'b')
plt.plot(t, gyros_y, 'r')
plt.plot(t, gyros_z, 'y')
plt.legend(['x', 'y', 'z'])
plt.grid()
plt.xlabel('Muestras')
plt.ylabel('Gyros (rad/s)')

plt.subplot(313)
plt.plot(t, hit_detection, 'g')
plt.legend(['x', 'y', 'z', 'hit'])
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Derivada')
plt.ylabel('Detección')

print("El numero de golpes es", hit_counter)
# Crear una figura para cada característica
plt.figure(figsize=(12, 8))

plt.subplot(231)
for hit_sample in hit_samples:
    plt.plot(hit_sample['accel_x'])

plt.grid()
plt.title('Accel_x')

plt.subplot(232)
for hit_sample in hit_samples:
    plt.plot(hit_sample['accel_y'])

plt.grid()
plt.title('Accel_y')

plt.subplot(233)
for hit_sample in hit_samples:
    plt.plot(hit_sample['accel_z'])

plt.grid()
plt.title('Accel_z')

plt.subplot(234)
for hit_sample in hit_samples:
    plt.plot(hit_sample['gyros_x'])

plt.grid()
plt.title('Gyros_x')

plt.subplot(235)
for hit_sample in hit_samples:
    plt.plot(hit_sample['gyros_y'])

plt.grid()
plt.title('Gyros_y')

plt.subplot(236)
for hit_sample in hit_samples:
    plt.plot(hit_sample['gyros_z'])

plt.grid()
plt.title('Gyros_z')

plt.show()




for idx, hit_sample in enumerate(hit_samples):
    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Golpe {idx + 1}')

    # Utilizar un índice fijo para representar las 50 muestras guardadas
    index_range = range(len(hit_sample['accel_x']))

    plt.subplot(231)
    plt.plot(index_range, hit_sample['accel_x'], 'b')
    plt.grid()
    plt.title('Accel_x')

    plt.subplot(232)
    plt.plot(index_range, hit_sample['accel_y'], 'r')
    plt.grid()
    plt.title('Accel_y')

    plt.subplot(233)
    plt.plot(index_range, hit_sample['accel_z'], 'y')
    plt.grid()
    plt.title('Accel_z')

    plt.subplot(234)
    plt.plot(index_range, hit_sample['gyros_x'], 'b')
    plt.grid()
    plt.title('Gyros_x')

    plt.subplot(235)
    plt.plot(index_range, hit_sample['gyros_y'], 'r')
    plt.grid()
    plt.title('Gyros_y')

    plt.subplot(236)
    plt.plot(index_range, hit_sample['gyros_z'], 'y')
    plt.grid()
    plt.title('Gyros_z')

    plt.show()
