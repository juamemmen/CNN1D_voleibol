#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 11:00:41 2024

@author: juanjo
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




#%%  Etiquetas de las actividades

LABELS = ['dedos',
          'saque',
          'antebrazos',
          'ataque']

# El número de pasos dentro de un segmento de tiempo
TIME_PERIODS = 50

# Los pasos a dar de un segmento al siguiente; si este valor es igual a
# TIME_PERIODS, entonces no hay solapamiento entre los segmentos
STEP_DISTANCE = 5

# al haber solapamiento aprovechamos más los datos

#%% cargamos los datos

column_names = ['tipo',
                'timestamp',
                    'accel_x',
                    'accel_y',
                    'accel_z',
                    'gyros_x',
                    'gyros_y',
                    'gyros_z']


df = pd.read_csv("datos_combinados_igual.csv", header=None,
                     names=column_names)


print(df.info())

#%% Datos que tenemos

print(df.shape)


#%% convertimos a flotante

def convert_to_float(x):
    try:
        return float(x)
    except:
        return np.nan



#%% Eliminamos entradas que contengan Nan --> ausencia de datos

df.dropna(axis=0, how='any', inplace=True)

#%% Mostramos los primeros datos

print(df.head())

#%% Mostramos los últimos

print(df.tail())

#%% Visualizamos la cantidad de datos que tenemos
# de cada actividad 

toque = df['tipo'].value_counts()
plt.bar(range(len(toque)), toque.values)
plt.xticks(range(len(toque)), toque.index)

#%% visualizamos 

def dibuja_datos_aceleracion(subset, actividad):
    plt.figure(figsize=(5,7))
    plt.subplot(311)
    plt.plot(subset["accel_x"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel X")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.title(actividad)
    plt.subplot(312)
    plt.plot(subset["accel_y"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Y")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.subplot(313)
    plt.plot(subset["accel_z"].values)
    plt.xlabel("Tiempo", fontsize=5)
    plt.ylabel("Acel Z")
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)

for actividad in np.unique(df['tipo']):
    subset = df[df['tipo'] == actividad][:50]
    dibuja_datos_aceleracion(subset, actividad)




#%% Normalizamos los datos

df["accel_x"] = (df["accel_x"] - min(df["accel_x"].values)) / (max(df["accel_x"].values) - min(df["accel_x"].values))
df["accel_y"] = (df["accel_y"] - min(df["accel_y"].values)) / (max(df["accel_y"].values) - min(df["accel_y"].values))
df["accel_z"] = (df["accel_z"] - min(df["accel_z"].values)) / (max(df["accel_z"].values) - min(df["accel_z"].values))
df["gyros_x"] = (df["gyros_x"] - min(df["gyros_x"].values)) / (max(df["gyros_x"].values) - min(df["gyros_x"].values))
df["gyros_y"] = (df["gyros_y"] - min(df["gyros_y"].values)) / (max(df["gyros_y"].values) - min(df["gyros_y"].values))
df["gyros_z"] = (df["gyros_z"] - min(df["gyros_z"].values)) / (max(df["gyros_z"].values) - min(df["gyros_z"].values))

#%% Representamos para ver que se ha hecho bien

plt.figure(figsize=(5,5))
plt.plot(df["accel_x"].values[:50])
plt.xlabel("Tiempo")
plt.ylabel("Acel X")

#%% Disión datos den entrenamiento y test

ventana=50


# Lista para almacenar los DataFrames de entrenamiento y prueba para cada tipo
df_train_list = []
df_test_list = []

# Especifica el porcentaje para entrenamiento
train_percentage = 0.8

# Itera sobre cada tipo presente en el conjunto de datos
for activity_type in np.unique(df['tipo']):
    
    # Filtra el DataFrame para el tipo actual
    df_activity_type = df[df['tipo'] == activity_type]
    
    # Calcula el índice para dividir los datos
    split_index = int(len(df_activity_type) * train_percentage)
    
    # Divide los datos en entrenamiento y prueba para el tipo actual
    df_train_type = df_activity_type.iloc[:split_index]
    df_test_type = df_activity_type.iloc[split_index:]
    
    # Almacena los DataFrames en las listas
    df_train_list.append(df_train_type)
    df_test_list.append(df_test_type)


df_train = pd.concat(df_train_list)
df_test = pd.concat(df_test_list)


#%% Codificamos la actividad de manera numérica

from sklearn import preprocessing

LABEL = 'ActivityEncoded'
# Transformar las etiquetas de String a Integer mediante LabelEncoder
le = preprocessing.LabelEncoder()

# Añadir una nueva columna al DataFrame existente con los valores codificados
df_train[LABEL] = le.fit_transform(df_train['tipo'].values.ravel())
df_test[LABEL] = le.fit_transform(df_test['tipo'].values.ravel())

print(df.head())
print(df_train[['tipo','ActivityEncoded']].value_counts())
print(df_test[['tipo','ActivityEncoded']].value_counts())

# Mostramos la información de los conjuntos de entrenamiento y prueba
print("Entrenamiento", df_train.shape)
print("Test", df_test.shape)

#%% comprobamos cual ha sido la división

print("Entrenamiento", df_train.shape[0]/df.shape[0])
print("Test", df_test.shape[0]/df.shape[0])

#%% Creamos las secuencias

from scipy import stats

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html

def create_segments_and_labels(df, time_steps, step, label_name):

    # x, y, z acceleraciones
    N_FEATURES = 6
    segments = []
    labels = []
    for i in range(0, len(df) - time_steps, step):
        xs = df['accel_x'].values[i: i + time_steps]
        ys = df['accel_y'].values[i: i + time_steps]
        zs = df['accel_z'].values[i: i + time_steps]
        xg = df['gyros_x'].values[i: i + time_steps]
        yg = df['gyros_y'].values[i: i + time_steps]
        zg = df['gyros_z'].values[i: i + time_steps]

        # Lo etiquetamos como la actividad más frecuente 
        label = stats.mode(df[label_name][i: i + time_steps])[0]
        segments.append([xs, ys, zs, xg, yg, zg])
        labels.append(label)

    # Los pasamos a vector
    reshaped_segments = np.asarray(segments, dtype= np.float32).reshape(-1, time_steps, N_FEATURES)
    labels = np.asarray(labels)

    return reshaped_segments, labels

x_train, y_train = create_segments_and_labels(df_train,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

x_test, y_test = create_segments_and_labels(df_test,
                                              TIME_PERIODS,
                                              STEP_DISTANCE,
                                              LABEL)

#%% observamos la nueva forma de los datos (50, 6)

print('x_train shape: ', x_train.shape)
print(x_train.shape[0], 'training samples')
print('y_train shape: ', y_train.shape)

#%% datos de entrada de la red neuronal

num_time_periods, num_sensors = x_train.shape[1], x_train.shape[2]
num_classes = le.classes_.size
print(list(le.classes_))

#%% transformamos los datos a flotantes

x_train = x_train.astype('float32')
#y_train = y_train.astype('float32')

x_test = x_test.astype('float32')
#y_test = y_test.astype('float32')

#%% Realizamos el one-hote econding para los datos de salida

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
y_train_hot = cat_encoder.fit_transform(y_train.reshape(len(y_train),1))
y_train = y_train_hot.toarray()

#%% RED NEURONAL

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Flatten
from keras.optimizers import Adam



filters = 64
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]


model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=5, activation='relu', input_shape=(n_timesteps, n_features)))
#model.add(Conv1D(filters=filters/2, kernel_size=5, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_outputs, activation='softmax'))
model.summary()

#%% Guardamos el mejor modelo y utilizamos early stopping

callbacks_list = [
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', save_best_only=True),
]

#%% determinamos la función de pérdida, optimizador y métrica de funcionamiento 
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping


optimizer = Adam(learning_rate=0.0001)
#optimizer = RMSprop(learning_rate=0.0001)
#optimizer='adam'
model.compile(loss='categorical_crossentropy',
                optimizer=optimizer, metrics=['accuracy'])


#%% Entrenamiento

BATCH_SIZE = 25
EPOCHS =50

#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(x_train,
                      y_train,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,# early_stopping],
                      validation_split=0.2,
                      verbose=1)

#%% Visualización entrenamiento

from sklearn.metrics import classification_report

plt.figure(figsize=(6, 4))
plt.plot(history.history['accuracy'], 'r', label='Accuracy of training data')
plt.plot(history.history['val_accuracy'], 'b', label='Accuracy of validation data')
plt.plot(history.history['loss'], 'r--', label='Loss of training data')
plt.plot(history.history['val_loss'], 'b--', label='Loss of validation data')
plt.title('Model Accuracy and Loss')
plt.ylabel('Accuracy and Loss')
plt.xlabel('Training Epoch')
plt.ylim(0)
plt.legend()
plt.show()

#%% Evaluamos el modelo en los datos de test

# actualizar dependiendo del nombre del modelo guardado
#model = keras.models.load_model("best_model.18-0.91.h5")

y_test_hot = cat_encoder.fit_transform(y_test.reshape(len(y_test),1))
y_test = y_test_hot.toarray()

test_loss, test_acc = model.evaluate(x_test, y_test)

print("Test accuracy", test_acc)
print("Test loss", test_loss)

#%%
# Print confusion matrix for training data
y_pred_train = model.predict(x_train)
# Take the class with the highest probability from the train predictions
max_y_pred_train = np.argmax(y_pred_train, axis=1)
max_y_train = np.argmax(y_train, axis=1)
print(classification_report(max_y_train, max_y_pred_train))

#%%
import seaborn as sns
from sklearn import metrics

def show_confusion_matrix(validations, predictions):

    matrix = metrics.confusion_matrix(validations, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(matrix,
                cmap='coolwarm',
                linecolor='white',
                linewidths=1,
                xticklabels=LABELS,
                yticklabels=LABELS,
                annot=True,
                fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

y_pred_test = model.predict(x_test)
# Toma la clase con la mayor probabilidad a partir de las predicciones de la prueba
max_y_pred_test = np.argmax(y_pred_test, axis=1)
max_y_test = np.argmax(y_test, axis=1)

show_confusion_matrix(max_y_test, max_y_pred_test)

print(classification_report(max_y_test, max_y_pred_test))