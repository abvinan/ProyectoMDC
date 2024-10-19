# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sG261uDkEwoGdZmdKwHxmxqeeVN5cGxZ
"""
import pandas as pd
import numpy as np
import streamlit as st
import implicit
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
import gdown
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar los datos desde Google Drive (archivo principal)
@st.cache_data
def cargar_datos():
    url = 'https://drive.google.com/uc?id=1NmAZBoSj8YqWFbypAm8HYMj2YHbRyggT'
    output = 'datos.csv'
    gdown.download(url, output, quiet=False)  # Descargar el archivo
    return pd.read_csv(output)

# Cargar el segundo archivo desde Google Drive (ventas mensuales)
@st.cache_data
def cargar_ventas_mensuales():
    url = 'https://drive.google.com/uc?id=1-21lc0LEqQLeph9YmnqIv5dhnDMzV15q'
    output = 'ventas_mensuales.csv'
    gdown.download(url, output, quiet=False)  # Descargar el archivo
    return pd.read_csv(output)

# 2. Filtrar productos por categoría seleccionada
def filtrar_por_categoria(df, categoria_seleccionada):
    secciones = {
        'Limpieza del Hogar': 14,
        'Cuidado Personal': 16,
        'Bebidas': 24,
        'Alimentos': 25
    }
    seccion = secciones.get(categoria_seleccionada)
    return df[df['SECCION'] == seccion]

# 3. Obtener el top 200 productos más vendidos de la categoría seleccionada
def obtener_top_200_productos(df_categoria):
    if df_categoria.empty:
        return pd.DataFrame()
    else:
        top_200_productos = df_categoria.groupby('COD_PRODUCTO')['CANTIDAD'].sum().nlargest(200).index
        return df_categoria[df_categoria['COD_PRODUCTO'].isin(top_200_productos)]

# 4. Obtener el top 5 de productos más vendidos de la categoría seleccionada (del segundo archivo)
def obtener_top_5_ventas_categoria(df_ventas, categoria_seleccionada):
    top_5_productos = df_ventas[df_ventas['Categoria'] == categoria_seleccionada].groupby('Código de Producto')['Cantidad Vendida'].sum().nlargest(5)
    return df_ventas[df_ventas['Código de Producto'].isin(top_5_productos.index)]

# 5. Preparar datos para entrenar el modelo ALS
def preparar_datos_para_entrenar(df):
    if len(df) > 0:
        df_train, df_test = train_test_split(df, test_size=0.3, random_state=42)
        df_train_compras = df_train.groupby(['COD_FACTURA', 'COD_PRODUCTO'])['CANTIDAD'].sum().unstack().fillna(0)
        df_test_compras = df_test.groupby(['COD_FACTURA', 'COD_PRODUCTO'])['CANTIDAD'].sum().unstack().fillna(0)
        return df_train_compras, df_test_compras
    else:
        st.error("No hay suficientes datos para dividir en entrenamiento y prueba.")
        return None, None

# 6. Entrenar el modelo ALS
def entrenar_modelo_als(df_train_compras):
    if df_train_compras is not None:
        df_train_sparse = csr_matrix(df_train_compras.values)
        als_model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
        als_model.fit(df_train_sparse)
        return als_model, df_train_sparse
    else:
        return None, None

# 7. Generar recomendaciones para el top 200 de productos y acumular resultados
def generar_recomendaciones_top_200(df_train_compras, als_model, df_train_sparse):
    if df_train_compras is not None and als_model is not None:
        als_recommendations = {}
        top_200_productos = df_train_compras.columns
        for product_id in top_200_productos:
            product_idx = df_train_compras.columns.get_loc(product_id)
            recommended_products = als_model.recommend(product_idx, df_train_sparse.T, N=6, filter_already_liked_items=False)
            recommended_products_list = []
            for i, score in zip(recommended_products[0], recommended_products[1]):
                if df_train_compras.columns[i] != product_id:
                    recommended_products_list.append(df_train_compras.columns[i])
            als_recommendations[product_id] = recommended_products_list[:5]
        return als_recommendations
    else:
        st.error("No se pudieron generar recomendaciones debido a la falta de datos o error en el entrenamiento.")
        return {}

# 8. Mostrar recomendaciones en formato tabla con descripción, precio, margen y promedio mensual de unidades vendidas

def mostrar_recomendaciones_tabla(product_id, als_recommendations, df, df_ventas):
    if product_id in als_recommendations:
        recomendaciones = als_recommendations[product_id]
        data = []

        for rec in recomendaciones:
            producto = df[df['COD_PRODUCTO'] == rec]
            if not producto.empty:
                descripcion = producto['DESC_PRODUCTO'].values[0]
                precio = producto['VALOR_PVSI'].values[0]
                costo = producto['COSTO'].values[0]
                margen = round(((precio - costo) / precio) * 100, 2)

                # Obtener el promedio mensual de ventas del producto recomendado y redondearlo al entero superior
                promedio_unidades = df_ventas[df_ventas['COD_PRODUCTO'] == rec]['Cantidad Vendida'].mean()
                promedio_unidades_redondeado = int(np.ceil(promedio_unidades))  # Redondear al entero superior

                data.append({
                    'Producto': descripcion, 
                    'Precio': f"${precio:.2f}", 
                    'Margen': f"{margen}%", 
                    'Promedio de unidades vendidas en el mes': promedio_unidades_redondeado  # Cambio en el nombre de la columna
                })
        
        # Convertir la lista de dicts en un DataFrame para mostrar en tabla verticalmente
        tabla = pd.DataFrame(data)
        
        # Ajustar los índices para que empiecen desde 1
        tabla.index = tabla.index + 1
        
        # Mostrar la tabla con un tamaño mayor y sin necesidad de scroll
        st.table(tabla)  # Cambiamos st.dataframe a st.table para que no haya expansión innecesaria y el tamaño se ajuste mejor
    else:
        st.write("No se encontraron recomendaciones para este producto.")

# 9. Visualización de gráficos: ventas mensuales, margen de ganancias, frecuencia de compra conjunta
def graficar_ventas_mensuales(df_ventas, productos_recomendados):
    fig, ax = plt.subplots(figsize=(10, 8))  # Tamaño reducido del gráfico
    # Cambiar 'Código de Producto' por 'COD_PRODUCTO'
    ventas = df_ventas[df_ventas['COD_PRODUCTO'].isin(productos_recomendados)].groupby(['MES', 'COD_PRODUCTO'])['Cantidad Vendida'].sum().unstack()
    
    # Mapeo de meses a abreviaturas en español
    meses_espanol = {1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'}
    
    # Asignar las abreviaciones a los ejes
    ventas.index = ventas.index.map(meses_espanol)
    
    # Crear gráfico apilado de ventas mensuales con colores en escala de naranja
    colores_naranja = ['#FFCC99', '#FF9966', '#FF6600', '#CC3300', '#993300']  # Colores en diferentes tonos de naranja
    ventas.plot(kind='bar', stacked=True, ax=ax, color=colores_naranja)
    
    # Cambiar el recuadro de leyenda para mostrar los números en vez de los códigos de productos
    handles, labels = ax.get_legend_handles_labels()
    labels = ['Producto {}'.format(i+1) for i in range(len(labels))]  # Cambiar los códigos por números de producto
    ax.legend(handles, labels, title='Productos Recomendados')
    
    ax.set_title("Ventas Mensuales por Producto Recomendado")
    ax.set_xlabel("MES")
    ax.set_ylabel("Unidades Vendidas")  # Cambio solicitado
    st.pyplot(fig)

def graficar_margen_ganancias(df_ventas, productos_recomendados):
    fig, ax = plt.subplots(figsize=(6, 4))  # Ajuste del tamaño
    
    # Filtrar por los productos recomendados
    margen_data = df_ventas[df_ventas['COD_PRODUCTO'].isin(productos_recomendados)].groupby('COD_PRODUCTO').mean()
    
    # Calcular el margen de ganancia: ((Precio Total - Costo Total) / Precio Total) * 100
    margen_data['Margen'] = ((margen_data['Precio Total'] - margen_data['Costo total']) / margen_data['Precio Total']) * 100
    
    # Crear el gráfico de barras con los márgenes
    ax.bar(margen_data.index.astype(str), margen_data['Margen'], color='skyblue')
    
    # Añadir etiquetas de valor a las barras
    for i, v in enumerate(margen_data['Margen']):
        ax.text(i, v + (2 if v > 0 else -2), f'{v:.2f}%', ha='center', color='black')
    
    ax.set_title("Margen de Ganancia por Producto")
    ax.set_xlabel("Producto")
    ax.set_ylabel("Margen (%)")
    plt.xticks(rotation=45, ha='right')  # Rotar los nombres para que sean legibles
    plt.tight_layout()  # Ajustar para que los elementos no se solapen
    st.pyplot(fig)


def graficar_frecuencia_compra_conjunta(df, productos_recomendados, producto_seleccionado):
    matriz_frecuencia = pd.crosstab(df['COD_FACTURA'], df['COD_PRODUCTO']).astype(bool).astype(int)
    productos_conjuntos = productos_recomendados + [producto_seleccionado]
    matriz_frecuencia = matriz_frecuencia[productos_conjuntos].T.dot(matriz_frecuencia[productos_conjuntos])
    
    fig, ax = plt.subplots()
    sns.heatmap(matriz_frecuencia, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title("Frecuencia de Compra Conjunta entre Productos")
    st.pyplot(fig)

# Configuración de la aplicación Streamlit
st.title("Sistema de Recomendación de Productos")

# Cargar datos
df = cargar_datos()
df_ventas = cargar_ventas_mensuales()

# Selección de categoría
categoria_seleccionada = st.sidebar.radio("Seleccione una Categoría", ('Limpieza del Hogar', 'Cuidado Personal', 'Bebidas', 'Alimentos'))

# Filtrar productos por categoría
df_categoria = filtrar_por_categoria(df, categoria_seleccionada)

if df_categoria.empty:
    st.warning(f"No se encontraron productos para la categoría seleccionada: {categoria_seleccionada}")
else:
    # Selección de subcategoría
    subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
    subcategoria_seleccionada = st.sidebar.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)

    # Filtrar productos por subcategoría
    df_subcategoria = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]
    productos_disponibles = df_subcategoria['DESC_PRODUCTO'].unique()

    # Selección de producto
    producto_seleccionado = st.sidebar.selectbox("Seleccione un Producto", productos_disponibles)
    producto_id = df_subcategoria[df_subcategoria['DESC_PRODUCTO'] == producto_seleccionado]['COD_PRODUCTO'].values[0]

    # Obtener el top 200 productos
    df_filtrado_top_200 = obtener_top_200_productos
    # Obtener el top 200 productos
    df_filtrado_top_200 = obtener_top_200_productos(df_categoria)

    if df_filtrado_top_200.empty:
        st.error("No se encontraron productos suficientes en la categoría para generar recomendaciones.")
    else:
        # Preparar datos para entrenar el modelo ALS
        df_train_compras, df_test_compras = preparar_datos_para_entrenar(df_filtrado_top_200)

        # Entrenar el modelo ALS
        als_model, df_train_sparse = entrenar_modelo_als(df_train_compras)

        # Generar las recomendaciones para el top 200 productos
        als_recommendations = generar_recomendaciones_top_200(df_train_compras, als_model, df_train_sparse)

        # Mostrar las recomendaciones en formato de tabla (productos recomendados horizontalmente)
        st.write("**Productos recomendados:**")
        mostrar_recomendaciones_tabla(producto_id, als_recommendations, df, df_ventas)

        # Obtener el top 5 de productos más vendidos en la categoría seleccionada
        def obtener_top_5_ventas_categoria(df_ventas, categoria_seleccionada):
            top_5_productos = df_ventas[df_ventas['Categoria'] == categoria_seleccionada].groupby('COD_PRODUCTO')['Cantidad Vendida'].sum().nlargest(5)
            return df_ventas[df_ventas['COD_PRODUCTO'].isin(top_5_productos.index)]      

        # Visualización de las ventas mensuales por producto recomendado
        productos_recomendados = als_recommendations.get(producto_id, [])
        st.write("**Gráfico de Ventas Mensuales por Producto Recomendado:**")
        graficar_ventas_mensuales(df_ventas, productos_recomendados)

        # Gráfico del margen de ganancias en relación a las ventas de los productos recomendados
        st.write("**Gráfico del Margen de Ganancia en Relación a las Ventas:**")
        graficar_margen_ganancias(df_ventas, productos_recomendados)

        # Gráfico de la frecuencia de compra conjunta (mapa de calor)
        st.write("**Mapa de Calor de Frecuencia de Compra Conjunta entre Productos:**")
        graficar_frecuencia_compra_conjunta(df, productos_recomendados, producto_id)

