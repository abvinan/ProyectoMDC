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
    df = pd.read_csv(output)

@st.cache_data
def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)
    df_ventas = pd.read_csv(output)

# Cargar datos
df = cargar_datos()
df_ventas = cargar_ventas_mensuales()

# Diccionario para mapear categorías con sus valores de sección
secciones = {
    'Limpieza del Hogar': 14,
    'Cuidado Personal': 16,
    'Bebidas': 24,
    'Alimentos': 25
}

# Función para filtrar productos por categoría seleccionada
def filtrar_por_categoria(df, categoria_seleccionada):
    seccion = secciones.get(categoria_seleccionada)
    return df[df['SECCION'] == seccion]

# Ventana de Selección: Categoría, Subcategoría y Productos
st.header("Selecciona los Productos para Recomendación")

# Selección de categoría basada en el diccionario de secciones
categoria_seleccionada = st.selectbox("Seleccione una Categoría", list(secciones.keys()))

# Filtrar el DataFrame para obtener solo los datos de la categoría seleccionada
df_categoria = filtrar_por_categoria(df, categoria_seleccionada)

# Filtrar subcategorías según la categoría seleccionada
subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
subcategoria_seleccionada = st.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)

# Filtrar productos según la subcategoría seleccionada
productos_disponibles = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()
productos_seleccionados = st.multiselect("Seleccione hasta 4 productos:", productos_disponibles, max_selections=4)

# Guardar la selección en el estado de sesión
if 'productos_seleccionados' not in st.session_state:
    st.session_state.productos_seleccionados = []
st.session_state.productos_seleccionados = productos_seleccionados

# Mostrar selección actual para verificar
st.write("Productos seleccionados:", st.session_state.productos_seleccionados)
