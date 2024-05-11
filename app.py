import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from inference import main as predict_risk

st.title("Прогностическая модель риска для здоровья беременных")
st.write("Добро пожаловать!")


# Отображение результатов
def show_results(probabilities):
    labels = ["Высокий риск", "Низкий риск", "Средний риск"]
    prediction = np.argmax(probabilities)
    st.write("#### Прогноз и его вероятность")
    if prediction == 0:
        st.error("Высокий риск")
    elif prediction == 1:
        st.success("Низкий риск")
    else:
        st.warning("Средний риск")

    # Визуализация вероятностей отнесения к классам
    fig, ax = plt.subplots()
    ax.barh(labels, probabilities[0])
    st.pyplot(fig)


# Основной код приложения
columns = st.columns(2)
with columns[0]:
    st.write("### Введите данные:")
    age = st.slider("Выберите возраст пациента", 18, 50, 23)
    systolic_bp = st.slider("Выберите верхнее давление", 80, 200, 90)
    diastolic_bp = st.slider("Выберите нижнее давление", 40, 120, 60)
    glucose_level = st.slider("Выберите уровень глюкозы в крови", 5.0, 25.0, 7.7)
    temperature = st.slider("Выберите температуру тела", 96.0, 115.0, 98.0)
    heart_rate = st.slider("Выберите ЧСС", 50, 120, 76)

with columns[1]:
    st.write("### Результаты:")
    result_container = st.empty()
    result_container.write(
        "Ожидаемый результат появится здесь после нажатия кнопки 'Рассчитать прогноз'"
    )
    button = st.button("Рассчитать прогноз")

if button:
    features = (
        age,
        systolic_bp,
        diastolic_bp,
        glucose_level,
        temperature,
        heart_rate,
    )
    probabilities = predict_risk(features=features)

    with columns[1]:
        result_container.write("Изменение параметров приведет к началу нового расчета")
        show_results(probabilities)
