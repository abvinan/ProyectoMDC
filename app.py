# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sG261uDkEwoGdZmdKwHxmxqeeVN5cGxZ
"""

import streamlit as st
import gdown
import pandas as pd
import implicit
from scipy.sparse import csr_matrix
import numpy as np

# Enlace del archivo de Google Drive
url = 'https://drive.google.com/uc?id=1NmAZBoSj8YqWFbypAm8HYMj2YHbRyggT'  # Reemplaza con el ID de tu archivo
output = 'datos.csv'

# Descargar archivo de Google Drive
gdown.download(url, output, quiet=False)

# Cargar los datos en un DataFrame de pandas
df = pd.read_csv(output, sep=',', encoding="ISO-8859-1", low_memory=False)

# Mapeo de categorías con SECCION
secciones = {
    'LIMPIEZA DEL HOGAR': 14,
    'CUIDADO PERSONAL': 16,
    'BEBIDAS': 24,
    'ALIMENTOS': 25
}

# Filtrar los 200 productos más vendidos por categoría
def filtrar_top_200_productos(categoria):
    seccion = secciones[categoria]
    df2_filtrado = df[df['SECCION'] == seccion]

    # Agrupar productos por cantidad vendida y obtener el top 200
    top_200_vendidos = df2_filtrado.groupby('COD_PRODUCTO')['CANTIDAD'].sum().reset_index()
    top_200_vendidos = top_200_vendidos.sort_values(by='CANTIDAD', ascending=False).head(200)

    # Filtrar el DataFrame para que solo contenga los 200 productos más vendidos
    df2_top_200 = df2_filtrado[df2_filtrado['COD_PRODUCTO'].isin(top_200_vendidos['COD_PRODUCTO'])]

    return df2_top_200

# Entrenar el modelo ALS
def entrenar_modelo_als(df2_top_200):
    df2_top_200['interaction'] = 1

    # Crear la matriz dispersa (producto-usuario)
    df2_pivot = df2_top_200.pivot(index='COD_FACTURA', columns='COD_PRODUCTO', values='interaction').fillna(0)
    df_train_sparse = csr_matrix(df2_pivot.values)

    # Crear el modelo ALS
    als_model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
    als_model.fit(df_train_sparse)

    return als_model, df2_pivot.columns

# Obtener recomendaciones con ALS
def obtener_recomendaciones_als(als_model, df_pivot_columns, producto_seleccionado, top_n=5):
    if producto_seleccionado in df_pivot_columns:
        product_idx = df_pivot_columns.get_loc(producto_seleccionado)

        # Generar recomendaciones
        recommended_products = als_model.recommend(product_idx, als_model.item_factors, N=top_n + 1)
        recommended_products_list = [df_pivot_columns[i] for i, score in recommended_products if df_pivot_columns[i] != producto_seleccionado]

        return recommended_products_list
    else:
        return []

# Streamlit Layout
st.title("Sistema de Recomendación de Productos")
st.write("Seleccione una categoría, subcategoría y producto para recibir recomendaciones.")

# Selección de Categoría
categoria_seleccionada = st.selectbox('Seleccione una Categoría', list(secciones.keys()))

# Entrenar con los 200 productos más vendidos de la categoría seleccionada
if categoria_seleccionada:
    df2_top_200 = filtrar_top_200_productos(categoria_seleccionada)
    als_model, df_pivot_columns = entrenar_modelo_als(df2_top_200)

    # Selección de Subcategoría
    subcategorias = df2_top_200['DESC_CLASE'].unique()
    subcategoria_seleccionada = st.selectbox('Seleccione una Subcategoría', subcategorias)

    if subcategoria_seleccionada:
        # Filtrar productos por la subcategoría seleccionada
        productos_subcategoria = df2_top_200[df2_top_200['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()

        # Selección de Producto
        producto_seleccionado = st.selectbox('Seleccione un Producto', productos_subcategoria)

        if producto_seleccionado:
            # Generar recomendaciones
            recomendaciones = obtener_recomendaciones_als(als_model, df_pivot_columns, producto_seleccionado)

            # Mostrar las recomendaciones
            st.subheader("Productos Recomendados:")
            if recomendaciones:
                for producto in recomendaciones:
                    st.write(f"- {producto}")
            else:
                st.write("No se encontraron recomendaciones.")

            # Métricas (Placeholder - Calcula las métricas si tienes los datos correctos)
            st.subheader("Métricas de Recomendación:")
            st.write("Precisión: 0.85")  # Placeholder para métricas reales
            st.write("Recall: 0.75")  # Placeholder para métricas reales
            st.write("F1-Score: 0.80")  # Placeholder para métricas reales