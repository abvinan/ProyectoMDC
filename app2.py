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

# Función para entrenar el modelo ALS
def entrenar_modelo_als(df):
    # Convertir datos a formato sparse matrix
    user_item_matrix = df.pivot(index='COD_FACTURA', columns='COD_PRODUCTO', values='CANTIDAD').fillna(0)
    sparse_matrix = csr_matrix(user_item_matrix.values)
    # Entrenar el modelo ALS
    modelo = AlternatingLeastSquares(factors=50, regularization=0.1, iterations=30)
    modelo.fit(sparse_matrix)
    return modelo, user_item_matrix

# Generar recomendaciones usando ALS
def generar_recomendaciones(modelo, user_item_matrix, producto_id, N=5):
    try:
        producto_idx = user_item_matrix.columns.get_loc(producto_id)
        recomendaciones = modelo.recommend(producto_idx, user_item_matrix.values.T, N=N, filter_already_liked_items=False)
        return [user_item_matrix.columns[i] for i, _ in recomendaciones]
    except KeyError:
        st.error(f"El producto con ID {producto_id} no se encontró en el modelo.")
        return []

# Función para calcular margen y precio de los combos
def calcular_combos(df, productos_seleccionados, als_recommendations):
    combos = []
    for producto_a in productos_seleccionados:
        for producto_b in als_recommendations.get(producto_a, []):
            prod_a_data = df[df['COD_PRODUCTO'] == producto_a]
            prod_b_data = df[df['COD_PRODUCTO'] == producto_b]
            
            if not prod_a_data.empty and not prod_b_data.empty:
                descripcion_a = prod_a_data['DESC_PRODUCTO'].values[0]
                descripcion_b = prod_b_data['DESC_PRODUCTO'].values[0]
                precio_a = prod_a_data['VALOR_PVSI'].values[0]
                costo_a = prod_a_data['COSTO'].values[0]
                precio_b = prod_b_data['VALOR_PVSI'].values[0]
                costo_b = prod_b_data['COSTO'].values[0]
                
                precio_combo = precio_a + precio_b
                margen_combo = round(((precio_combo - (costo_a + costo_b)) / precio_combo) * 100, 2)
                
                combos.append({
                    'Producto A': descripcion_a,
                    'Producto B': descripcion_b,
                    'Precio Combo': precio_combo,
                    'Margen Combo': f"{margen_combo}%",
                    'Check': False
                })
    return pd.DataFrame(combos)

# Función para calcular las métricas de la tercera ventana
def calcular_resumen_combos(df_ventas, combos_seleccionados):
    resumen = []
    for _, combo in combos_seleccionados.iterrows():
        prod_a_data = df_ventas[df_ventas['DESC_PRODUCTO'] == combo['Producto A']]
        prod_b_data = df_ventas[df_ventas['DESC_PRODUCTO'] == combo['Producto B']]
        
        if not prod_a_data.empty and not prod_b_data.empty:
            cantidad_a = prod_a_data['Cantidad Vendida'].mean()
            cantidad_b = prod_b_data['Cantidad Vendida'].mean()
            precio_total_a = prod_a_data['Precio Total'].mean()
            precio_total_b = prod_b_data['Precio Total'].mean()
            costo_total_a = prod_a_data['Costo total'].mean()
            costo_total_b = prod_b_data['Costo total'].mean()
            
            cantidad_venta_estimada = int(np.ceil(cantidad_a + cantidad_b))
            venta_estimada = round(precio_total_a + precio_total_b, 2)
            ganancia_estimada = round((venta_estimada - (costo_total_a + costo_total_b)), 2)
            
            resumen.append({
                'Producto A': combo['Producto A'],
                'Producto B': combo['Producto B'],
                'Cantidad estimada de venta': cantidad_venta_estimada,
                'Venta estimada ($)': f"${venta_estimada}",
                'Ganancia estimada ($)': f"${ganancia_estimada}"
            })
    return pd.DataFrame(resumen)

# Configuración de la aplicación Streamlit
st.title("Sistema de Recomendación de Productos")
menu_seleccion = st.sidebar.radio("Seleccione una ventana:", ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"])

# Ventana 1: Selección de Productos con Categoría y Subcategoría
if menu_seleccion == "Seleccionar Productos":
    st.header("Selecciona los Productos para Recomendación")

    # Selección de categoría
    categorias_disponibles = df['Categoria'].unique()
    categoria_seleccionada = st.selectbox("Seleccione una Categoría", categorias_disponibles)
    
    # Filtrar subcategorías según la categoría seleccionada
    subcategorias_disponibles = df[df['Categoria'] == categoria_seleccionada]['DESC_CLASE'].unique()
    subcategoria_seleccionada = st.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)

    # Filtrar productos según la subcategoría seleccionada
    productos_disponibles = df[(df['Categoria'] == categoria_seleccionada) & 
                               (df['DESC_CLASE'] == subcategoria_seleccionada)]['DESC_PRODUCTO'].unique()
    productos_seleccionados = st.multiselect("Selecciona hasta 4 productos:", productos_disponibles, max_selections=4)
    
    # Guardar la selección en el estado de sesión
    if 'productos_seleccionados' not in st.session_state:
        st.session_state.productos_seleccionados = []
    st.session_state.productos_seleccionados = productos_seleccionados

# Ventana 2: Mostrar Combos Recomendados
elif menu_seleccion == "Recomendaciones":
    st.header("Combos Recomendados")

    if st.session_state.get('productos_seleccionados'):
        # Entrenar el modelo ALS y generar recomendaciones
        modelo, user_item_matrix = entrenar_modelo_als(df)
        als_recommendations = {}

        for producto in st.session_state.productos_seleccionados:
            producto_id = df[df['DESC_PRODUCTO'] == producto]['COD_PRODUCTO'].values[0]  # Obtener el ID del producto
            als_recommendations[producto] = generar_recomendaciones(modelo, user_item_matrix, producto_id)

        # Crear el dataframe de combos
        df_combos = calcular_combos(df, st.session_state.productos_seleccionados, als_recommendations)

        # Mostrar tabla de combos con opción de selección (checkbox)
        st.write("Seleccione los combos que desea incluir en el resumen:")

        # Agregar checkboxes para selección de combos
        seleccionados = []
        for i, row in df_combos.iterrows():
            if st.checkbox(f"Seleccionar Combo {i+1}: {row['Producto A']} + {row['Producto B']}", key=f"combo_{i}"):
                seleccionados.append(row)

        # Guardar los combos seleccionados en st.session_state
        if seleccionados:
            st.session_state.combos_seleccionados = pd.DataFrame(seleccionados)
        else:
            st.session_state.combos_seleccionados = pd.DataFrame()

# Ventana 3: Resumen de Combos Seleccionados
elif menu_seleccion == "Resumen de Combos Seleccionados":
    st.header("Resumen de Combos Seleccionados")

    # Verificamos que 'combos_seleccionados' esté definido y no esté vacío
    if 'combos_seleccionados' in st.session_state and not st.session_state.combos_seleccionados.empty:
        # Calculamos y mostramos el resumen de combos seleccionados
        resumen_df = calcular_resumen_combos(df_ventas, st.session_state.combos_seleccionados)
        st.table(resumen_df)
    else:
        st.write("No se han seleccionado combos.")

