import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# Mostrar o cargar los datos
#ds = pd.read_csv("dataset_financiero_riesgo.csv")

# Colocar un titulo principal en la página Web
st.title("Predicción de Riesgo Financiero")

# Cargar los datos en la memoria CACHE para mejorar la velocidad del acceso al 
# conjunto de datos
@st.cache_data 

# Hacemos una funcion que se llama cargar_datos. Leemos el archivo en una variable
# y retornamos la variable al que llama a la función. En este caso la variable 
# que retornamos se llama "ds", abreviatura de "dataset".
def cargar_datos():
    ds = pd.read_csv("dataset_financiero_riesgo.csv")
    return ds

ds = cargar_datos()
st.write("Vista previa de los datos")
st.dataframe(ds.head())

#PREPOROCESAMIENTO DE DATOS O DEL CONJUNTO DE DATOS
ds_encode = ds.copy()