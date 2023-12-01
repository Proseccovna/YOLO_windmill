import torch
import streamlit as st
import io
import imageio
import requests
from PIL import Image
from torchvision import transforms as T
from torchvision.models import densenet121
# from torchvision import io
import torch.nn as nn
import time
from model.predict import predict_1
from yolo_model.yolo8 import detect
import tempfile



# st.set_page_config(layout='wide')

st.title('Компьютерное зрение • Computer Vision')


with st.sidebar:
    st.header('Выберите страницу')
    page = st.selectbox("Выберите страницу", ["Главная", "Ветряные мельницы", "Текст", "Итоги"])

if page == "Главная":
    st.header('Выполнила команда "YOLO":')
    st.subheader('🦁Алексей Д.')
    st.subheader('🐱Ерлан')
    st.subheader('🐰Тата')
    st.subheader('🐯Тигран')

    st.header(" 🌟 " * 10)

    st.header('Наши задачи:')
    st.subheader('*Задача №1*: Детектирование ветряных мельниц')
    st.subheader('*Задача №2*: Очистка текста от зашумлений')


elif page == "Ветряные мельницы":
    st.header("Процесс обучения:")
    st.subheader("- Модель: *YOLOv8 Nano*")
    st.subheader("- Количество эпох обучения: *64*")
    st.subheader("- mAP50: *~ 0.83*")

    st.info('Расширение картинки должно быть в формате .jpg /.jpeg /.png')
    image_url = st.text_input("Введите URL изображения")
    start_time = time.time()

    if image_url:

        response = requests.get(image_url)
        image = Image.open(io.BytesIO(response.content))
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            image.save(temp_image.name)
        st.subheader('Ваше фото до детекции:')
        st.image(image, caption='Original Image', use_column_width=True)

        show_result_button1 = st.button("Показать результат", key="result_button_1")

        if show_result_button1:
            st.success("Ваш результат готов!")
            st.subheader('Предсказание модели:')
            detection_result = detect(temp_image.name)
    # Выведение второй картинки с нарисованными рамками
            st.image(detection_result, caption='Image with Detection Result', use_column_width=True)
            st.subheader(f'Время предсказания: {round((time.time() - start_time), 2)} сек.')
            st.header('🎈' * 10)

    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    start_time_file = time.time()
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_image:
            uploaded_image = Image.open(uploaded_file)
            uploaded_image.save(temp_image.name)
        st.subheader('Ваше фото до детекции:')
        st.image(uploaded_image, caption='Original Image', use_column_width=True)

        show_result_button2 = st.button("Показать результат", key="result_button_2")
        if show_result_button2:
            st.success("Ваш результат готов!")

            st.subheader("Предсказание модели:")
            prediction_result_file = detect(temp_image.name)
            st.image(prediction_result_file, caption='Image with Detection Result', use_column_width=True)
            st.subheader(f'Время предсказания: {round((time.time() - start_time_file), 2)} сек.')
            st.header('🎈' * 10)


elif page == "Текст":
    st.header("Процесс обучения:")

    st.subheader("- Модель: *ConvAutoencoder()*")
    st.subheader("- Количество эпох обучения: *100*")

    st.info('Расширение картинки должно быть в формате .jpg /.jpeg /.png')
    image_url2 = st.text_input("Введите URL изображения")
    start_time2 = time.time()

    if image_url2:
        # Загрузка изображения по ссылке
        response2 = requests.get(image_url2)
        image2 = Image.open(io.BytesIO(response2.content))
        st.subheader('Ваше фото до обработки:')
        st.image(image2)
        prediction_result = predict_1(image2)

        show_result_button3 = st.button("Показать результат", key="result_button_3")
        if show_result_button3:
            st.success("Ваш результат готов!")

            st.subheader("Ваше фото после обработки:")
            st.image(prediction_result, channels='GRAY')
            st.subheader(f'Время предсказания: {round((time.time() - start_time2), 2)} сек.')
            st.header('🎈' * 10)

    uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
    start_time_file = time.time()

    if uploaded_file is not None:
        image_file = Image.open(uploaded_file)
        st.subheader('Ваше фото до обработки:')
        st.image(image_file)
        prediction_result_file = predict_1(image_file)

        show_result_button4 = st.button("Показать результат", key="result_button_4")
        if show_result_button4:
            st.success("Ваш результат готов!")
            st.subheader("Ваше фото после обработки:")
            st.image(prediction_result_file, channels='GRAY')
            st.subheader(f'Время предсказания: {round((time.time() - start_time_file), 2)} сек.')
            st.header('🎈' * 10)



elif page == "Итоги":
    st.header('Результаты и выводы')
    st.subheader('*Задача №1*: Детектирование ветряных мельниц')

    st.subheader("Метрики из Clear ML")
    image_1 = Image.open("pictures/P_curve.png")
    image_2 = Image.open("pictures/PR_curve.png")
    image_3 = Image.open("pictures/R_curve.png")
    image_4 = Image.open("pictures/F1_curve.png")

# Отображаем изображения в одной строке
    st.image([image_1, image_2, image_3, image_4], caption=['Image 1 - P_curve', 'Image 2 - PR_curve', 'Image 3 - R_curve', 'Image 4 - F1_curve'], width=300)

    st.subheader("Результативные графики из Clear ML")
    image_5 = imageio.imread('pictures/plots.jpg')[:, :, :]
    st.image(image_5)
    st.subheader("Обучение")
    image_6 = imageio.imread('pictures/train.png')[:, :, :]
    st.image(image_6)

    st.subheader("Confusion matrix")

    image_7 = imageio.imread("pictures/confusion_matrix.png")[:, :, :]
    st.image(image_7)

    st.subheader("Confusion matrix normolized")
    
    image_8 = imageio.imread('pictures/confusion_matrix_normalized.png')[:, :, :]
    st.image(image_8)

    st.subheader("Еще мы пробовали *YOLOv8 Medium* на 30 эпохах:")
    st.markdown('YOLOv8 Medium')

    image_9 = imageio.imread('pictures/pt1-3.png')[:, :, :]
    st.image(image_9)

    st.markdown('YOLOv8 Nano')
    
    image_10 = imageio.imread('pictures/pt1-2.png')[:, :, :]
    st.image(image_10)

    st.markdown('YOLOv8 Medium')

    image_11 = imageio.imread('pictures/pt2-3.png')[:, :, :]
    st.image(image_11)

    st.markdown('YOLOv8 Nano')
    
    image_12 = imageio.imread('pictures/pt2-2.png')[:, :, :]
    st.image(image_12)


    st.subheader('*Задача №2*: Очистка текста от зашумлений')
    st.subheader("Используемая модель:")
    st.subheader("*criterion = nn.L1Loss()*")
    st.subheader('*optimizer = torch.optim.SGD(model.parameters(), lr=0.005)")*')
    image_9 = imageio.imread('pictures/model.png')[:, :, :]
    st.image(image_9)
