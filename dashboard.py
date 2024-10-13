# -*- coding: utf-8 -*-
"""dashboard.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/18HfzUkdRBYqOcv7fofhiIXbsXU379HT3
"""

import streamlit as st
import pandas as pd
import plotly.express as px

# Título de la aplicación
st.title('Dashboard Interactivo de Ventas')

# Datos de ejemplo
df = pd.DataFrame({
    'Producto': ['Producto A', 'Producto B', 'Producto C'],
    'Ventas': [30, 45, 25]
})

# Dropdown para seleccionar el producto
producto_seleccionado = st.selectbox('Selecciona un producto', df['Producto'])

# Filtrar los datos según el producto seleccionado
datos_filtrados = df[df['Producto'] == producto_seleccionado]

# Crear gráfico usando Plotly
fig = px.bar(datos_filtrados, x='Producto', y='Ventas', title=f'Ventas de {producto_seleccionado}')

# Mostrar gráfico
st.plotly_chart(fig)