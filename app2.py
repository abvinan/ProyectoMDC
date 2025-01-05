import streamlit as st
import pandas as pd
import numpy as np
import gdown
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from sklearn.model_selection import train_test_split

# AUTENTICACIÓN
USER_CREDENTIALS = {"username": "admin", "password": "password123"}

# CSS para ajustar el diseño de la ventana de inicio de sesión
st.markdown("""
    <style>
    body {
        margin: 0;
        padding: 0;
        background-color: #f4f4f4;
    }
    .main-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
    }
    .login-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        width: 320px;
        text-align: left;
    }
    .login-box h1 {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 15px;
        color: #333;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Función para manejar la autenticación
def autenticar_usuario():
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False

    if not st.session_state["autenticado"]:
        st.markdown('<div class="main-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<h1>Iniciar Sesión</h1>', unsafe_allow_html=True)

        # Capturar credenciales de usuario y contraseña
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")

        # Botón para autenticar
        if st.button("Iniciar Sesión"):
            if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
                st.session_state["autenticado"] = True
                st.success("Inicio de sesión exitoso. Redirigiendo...")
            else:
                st.error("Usuario o contraseña incorrectos.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Función para cargar datos
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

def sistema_recomendacion():
    st.title("Bienvenido a la Aplicación de Recomendación")
    st.write("¡La aplicación está funcionando correctamente!")

    # Menú lateral
    menu_seleccion = st.sidebar.radio(
        "Seleccione una ventana:", 
        ["Seleccionar Productos", "Recomendaciones", "Resumen de Combos Seleccionados"]
    )
    
    # Ventana 1: Selección de Productos
    if menu_seleccion == "Seleccionar Productos":
        st.header("Selecciona los Productos para Recomendación")
        categoria_seleccionada = st.selectbox("Seleccione una Categoría", list(secciones.keys()))
        df_categoria = filtrar_por_categoria(df, categoria_seleccionada)
        st.session_state['df_categoria'] = df_categoria
        subcategorias_disponibles = df_categoria['DESC_CLASE'].unique()
        subcategoria_seleccionada = st.selectbox("Seleccione una Subcategoría", subcategorias_disponibles)
        productos_disponibles = df_categoria[df_categoria['DESC_CLASE'] == subcategoria_seleccionada]['DESC_PRODUCTO'].unique()
        productos_seleccionados = st.multiselect("Seleccione hasta 4 productos:", productos_disponibles, max_selections=4)
        st.session_state.productos_seleccionados = productos_seleccionados

    # Ventana 2: Mostrar Combos Recomendados
    elif menu_seleccion == "Recomendaciones":
        st.header("Combos Recomendados")
        if 'productos_seleccionados' in st.session_state and st.session_state.productos_seleccionados:
            df_categoria = st.session_state['df_categoria']
            productos_seleccionados_ids = [df[df['DESC_PRODUCTO'] == nombre]['COD_PRODUCTO'].values[0] for nombre in st.session_state.productos_seleccionados]
            df_top_200 = obtener_top_200_productos(df_categoria)
            df_train_compras = preparar_datos_para_entrenar(df_top_200)
            modelo_als, df_train_sparse = entrenar_modelo_als(df_train_compras)
            recomendaciones = generar_recomendaciones_seleccionados(df_train_compras, modelo_als, df_train_sparse, productos_seleccionados_ids)
            combos = []
            for producto_id in productos_seleccionados_ids:
                descripcion_a = df[df['COD_PRODUCTO'] == producto_id]['DESC_PRODUCTO'].values[0]
                for recomendacion_id in recomendaciones.get(producto_id, []):
                    descripcion_b = df[df['COD_PRODUCTO'] == recomendacion_id]['DESC_PRODUCTO'].values[0]
                    precio_a = df[df['COD_PRODUCTO'] == producto_id]['VALOR_PVSI'].values[0]
                    precio_b = df[df['COD_PRODUCTO'] == recomendacion_id]['VALOR_PVSI'].values[0]
                    costo_a = df[df['COD_PRODUCTO'] == producto_id]['COSTO'].values[0]
                    costo_b = df[df['COD_PRODUCTO'] == recomendacion_id]['COSTO'].values[0]
                    precio_combo = precio_a + precio_b
                    margen_combo = round(((precio_combo - (costo_a + costo_b)) / precio_combo) * 100, 2)
                    combos.append({
                        'Producto A': descripcion_a,
                        'Producto B': descripcion_b,
                        'Precio Combo': f"${precio_combo:.2f}",
                        'Margen Combo': f"{margen_combo}%"
                    })
            df_combos = pd.DataFrame(combos)
            st.table(df_combos)
            seleccion_indices = st.multiselect("Seleccione los índices de los combos que desea considerar:", df_combos.index.tolist())
            st.session_state.combos_seleccionados = df_combos.loc[seleccion_indices]

    # Ventana 3: Resumen de Combos Seleccionados
    elif menu_seleccion == "Resumen de Combos Seleccionados":
        st.header("Resumen de Combos Seleccionados")
        
        if 'combos_seleccionados' in st.session_state and not st.session_state.combos_seleccionados.empty:
            resumen = []
            for _, row in st.session_state.combos_seleccionados.iterrows():
                producto_a = row['Producto A']
                producto_b = row['Producto B']
                
                # Obtener los códigos de producto correspondientes
                try:
                    producto_a_id = df[df['DESC_PRODUCTO'] == producto_a]['COD_PRODUCTO'].values[0]
                    producto_b_id = df[df['DESC_PRODUCTO'] == producto_b]['COD_PRODUCTO'].values[0]
                except IndexError:
                    st.error(f"No se encontró el código para '{producto_a}' o '{producto_b}'.")
                    continue  # Saltar este combo si falta alguno de los códigos
    
                # Filtrar las ventas mensuales de cada producto usando los nombres de columna correctos
                ventas_a = df_ventas[df_ventas['COD_PRODUCTO'] == producto_a_id]
                ventas_b = df_ventas[df_ventas['COD_PRODUCTO'] == producto_b_id]
    
                # Verificar si hay datos disponibles para ambos productos
                if not ventas_a.empty and not ventas_b.empty:
                    # Calcular las métricas requeridas utilizando los nombres de columna correctos y formatear con separadores de miles
                    cantidad_estimada = "{:,.0f}".format(ventas_a['Cantidad Vendida'].mean() + ventas_b['Cantidad Vendida'].mean())
                    venta_estimada = "{:,.0f}".format(ventas_a['Precio Total'].mean() + ventas_b['Precio Total'].mean())
                    ganancia_estimada = "{:,.0f}".format(
                        (ventas_a['Precio Total'].mean() - ventas_a['Costo total'].mean()) +
                        (ventas_b['Precio Total'].mean() - ventas_b['Costo total'].mean())
                    )
    
                    # Agregar los datos del combo al resumen
                    resumen.append({
                        'Combo': f"{producto_a} + {producto_b}",
                        'Cantidad estimada de venta': cantidad_estimada,
                        'Venta estimada ($)': venta_estimada,
                        'Ganancia estimada ($)': ganancia_estimada
                    })
                else:
                    st.warning(f"No hay datos de ventas mensuales para '{producto_a}' o '{producto_b}'.")
    
            # Mostrar el resumen si se generaron datos
            if resumen:
                df_resumen = pd.DataFrame(resumen)
                st.table(df_resumen)
            else:
                st.write("No se han generado datos de resumen para los combos seleccionados.")
        else:
            st.write("No se han seleccionado combos para mostrar el resumen.")

# Declaración de la variable secciones
secciones = {
    'Limpieza del Hogar': 14,
    'Cuidado Personal': 16,
    'Bebidas': 24,
    'Alimentos': 25
}

# Lógica principal
if "autenticado" not in st.session_state or not st.session_state["autenticado"]:
    autenticar_usuario()
else:
    sistema_recomendacion()

