import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

# Funciones para cargar datos
@st.cache_data
def cargar_datos():
    url = 'https://drive.google.com/uc?id=1NmAZBoSj8YqWFbypAm8HYMj2YHbRyggT'
    output = 'datos.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

@st.cache_data
def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)
    return pd.read_csv(output)

# Cargar datos
df = cargar_datos()
df_ventas = cargar_ventas_mensuales()

# Mostrar las primeras filas y las columnas de los DataFrames para verificar la estructura
st.write("Estructura del DataFrame 'datos':")
st.write(df.head())
st.write("Columnas en 'datos':", df.columns)

st.write("Estructura del DataFrame 'ventas_mensuales':")
st.write(df_ventas.head())
st.write("Columnas en 'ventas_mensuales':", df_ventas.columns)
