import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
def main():

    ## pasos inciales de diseño
    st.title("Solución del project Pagina Web de ML con Streamlit Data Science")
    st.header("esto es un encabezado ")
    st.subheader("Esto en un subencabezado ")
    st.text("esto es un texto")

    nombre = "Giussepe"

    st.text(f"Mi nombre es {nombre} y soy alumno")
    st.success("Este es mi mensaje de de aprobado con exito")
    st.warning("No se visualizar el project corregir y enviar")
    st.info("Tarea rechazada, vuelva a enviar")

    ## iniciar con el modelo

    # datos dummies

    np.random.seed(42)
    X = np.random.rand(100,1)*10
    y = 3* X +8 + np.random.randn(100,1)*2

    # separar conjunto de datos entre entrenamiento y test
    X_train , X_test, y_train, y_test = train_test_split(X,y, test_size= 0.3, random_state=42)

    #generar el modelo vamos a usar regresion lineal
    model = LinearRegression()
    model.fit(X_train,y_train)

    #prediccion
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred)

    ## la interfaz
    st.title("Mi primer regresión lineal en web ")
    st.write(" este es un modelo para entregar el project")

    # usar un SelectBox
    opcion = st.selectbox("Seleccione el tipo de visualización", ["Dispersión", "Línea de Regresión"])

    #checkbox para mostar coeficientes
    if st.checkbox("Mostrar coeficientes de la Regresión Líneal"):
        st.write(f"coeficiente:{model.coef_[0][0]:.2f}")
        st.write(f"coeficiente intersección:{model.intercept_[0]:.2f}")
        st.write(f"Error medio cuadratico: {mse:.2f}")

    #slider
    data_range = st.slider("Seleccione el rago que quiere evaluar",0,100,(10,90) )
    x_display = X_test[data_range[0]:data_range[1]]
    y_display = y_test[data_range[0]:data_range[1]]
    y_pred_display = y_pred[data_range[0]:data_range[1]]

if __name__ == '__main__':
    main()