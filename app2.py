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
    st.write("Datos cargados desde 'datos.csv':")
    st.write(df.head())  # Mostrar las primeras filas para verificar la estructura
    return df

@st.cache_data
def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)
    df_ventas = pd.read_csv(output)
    st.write("Datos cargados desde 'ventas_mensuales.csv':")
    st.write(df_ventas.head())  # Mostrar las primeras filas para verificar la estructura
    return df_ventas

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

# Función para preparar la matriz dispersa y entrenar el modelo ALS
def entrenar_modelo_als(df):
    # Verificar si las columnas existen
    if not {'COD_FACTURA', 'COD_PRODUCTO', 'CANTIDAD'}.issubset(df.columns):
        st.error("Las columnas necesarias no están en el DataFrame. Verifique que 'COD_FACTURA', 'COD_PRODUCTO' y 'CANTIDAD' existan en 'datos.csv'.")
        return None, None
    
    # Crear matriz dispersa con facturas en las filas y productos en las columnas
    user_item_matrix = df.pivot(index='COD_FACTURA', columns='COD_PRODUCTO', values='CANTIDAD').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)

    # Agregar impresión de depuración
    st.write("User-Item Matrix (Sparse):")
    st.write(user_item_matrix.head())

    # Entrenar el modelo ALS
    modelo = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
    modelo.fit(sparse_matrix)

    # Agregar impresión de depuración
    st.write("Modelo ALS entrenado correctamente")

    # Devolver el modelo y la matriz para su uso posterior
    return modelo, user_item_matrix

# Entrenar el modelo ALS
modelo, user_item_matrix = entrenar_modelo_als(df)

# Verificar que el modelo y la matriz se hayan creado correctamente
if modelo is not None and user_item_matrix is not None:
    # Función para generar recomendaciones basadas en el modelo ALS
    def generar_recomendaciones(modelo, user_item_matrix, producto_id, N=5):
        try:
            # Obtener el índice del producto en la matriz
            producto_idx = user_item_matrix.columns.get_loc(producto_id)
            st.write(f"Índice del producto {producto_id}: {producto_idx}")
            
            # Generar recomendaciones
            recomendaciones = modelo.recommend(producto_idx, user_item_matrix.values.T, N=N, filter_already_liked_items=False)
            
            # Mostrar recomendaciones generadas para depuración
            st.write("Recomendaciones generadas:", recomendaciones)
            
            # Devolver los códigos de producto recomendados
            return [user_item_matrix.columns[i] for i, _ in recomendaciones]
        except KeyError:
            st.error(f"El producto con ID {producto_id} no se encontró en el modelo.")
            return []
        except Exception as e:
            st.error(f"Error en la función de recomendaciones: {e}")
            return []

    # Generar recomendaciones para los productos seleccionados por el usuario
    st.write("Recomendaciones de productos:")
    for producto in st.session_state.productos_seleccionados:
        # Obtener el código de producto (COD_PRODUCTO) correspondiente al nombre del producto seleccionado
        producto_id = df[df['DESC_PRODUCTO'] == producto]['COD_PRODUCTO'].values[0]
        
        # Generar recomendaciones usando el modelo
        recomendaciones = generar_recomendaciones(modelo, user_item_matrix, producto_id)
        
        # Mostrar recomendaciones para cada producto seleccionado
        st.write(f"Para el producto '{producto}' (ID: {producto_id}), se recomiendan:")
        st.write(recomendaciones)
else:
    st.write("No se pudo entrenar el modelo ALS debido a problemas en la estructura de datos.")
