import os
import cv2
import numpy as np


# Функция для получения всех изображений из указанной директории
def get_all_img_src(img_dir):
    valid_extensions = {".png", ".jpg", ".jpeg"}
    return [
        os.path.join(img_dir, f)
        for f in os.listdir(img_dir)
        if os.path.splitext(f)[1].lower() in valid_extensions
    ]


# Функция для загрузки модели YOLO
def load_yolo(weights_path, config_path, names_path):
    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return net, classes, output_layers


# Функция для детекции объектов на изображении
def detect_objects(img, net, output_layers):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    return outs, width, height


# Функция для получения координат и размеров боксов с автомобилями
def get_boxes(
    outs,
    width,
    height,
    classes,
    target_class="car",
    confidence_threshold=0.5,
    nms_threshold=0.4,
):
    boxes = []
    confidences = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold and classes[class_id] == target_class:
                center_x, center_y = int(detection[0] * width), int(
                    detection[1] * height
                )
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                boxes.append((x, y, w, h))
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    if len(indices) > 0 and isinstance(indices[0], (list, np.ndarray)):
        filtered_boxes = [boxes[i[0]] for i in indices]
    else:
        filtered_boxes = [boxes[i] for i in indices]
    return filtered_boxes


# Функция для определения среднего цвета автомобиля
def get_car_color(img, box):
    x, y, w, h = box
    x = max(0, x)
    y = max(0, y)
    w = min(w, img.shape[1] - x)
    h = min(h, img.shape[0] - y)
    car_region = img[y : y + h, x : x + w]
    avg_color = np.mean(car_region, axis=(0, 1))[::-1]  # конвертируем BGR в RGB
    return tuple(int(c) for c in avg_color)


# Основная функция для обработки всех изображений в директории
def process_images(img_dir, net, classes, output_layers):
    all_img_src = get_all_img_src(img_dir)
    results = []
    for img_src in all_img_src:
        image = cv2.imread(img_src)
        outs, width, height = detect_objects(image, net, output_layers)
        boxes = get_boxes(outs, width, height, classes)
        car_colors = [get_car_color(image, box) for box in boxes]
        results.append((img_src, car_colors))
    return results


# Главная функция для запуска
def main(img_dir, weights_path, config_path, names_path):
    net, classes, output_layers = load_yolo(weights_path, config_path, names_path)
    results = process_images(img_dir, net, classes, output_layers)
    for img_src, car_colors in results:
        print(f"Colors for {img_src}: {car_colors}")


if __name__ == "__main__":
    # Параметры для запуска
    img_dir = "cars_selected"  # взят датасет https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset (оставлено 10 штук из-за большого размера)
    weights_path = "model/yolov3.weights"
    config_path = "model/yolov3.cfg"
    names_path = "model/coco.names"

    # Запуск
    main(img_dir, weights_path, config_path, names_path)
