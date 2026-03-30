from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import pandas as pd
import argparse

# ПАРСИНГ АРГУМЕНТОВ КОМАНДНОЙ СТРОКИ
parser = argparse.ArgumentParser(description="Скрипт обработки видео")
parser.add_argument('--video', type=str, default='видео 2.mp4', help='Путь к видеофайлу')
args = parser.parse_args()
input_video_path = args.video # Путь к входному видео

# КОНФИГУРАЦИЯ ПРОЕКТА

# Загрузка модели YOLOv8
model = YOLO('yolov8n.pt')

output_video_path = 'output.mp4' # Путь для сохранения обработанного видео

# Параметры гистерезиса (чтобы избежать ложных срабатываний)
absence_counter = 0  # Счетчик кадров отсутствия
presence_counter = 0 # Счетчик кадров присутствия
REQUIRED_PRESENT_FRAMES = 15  # Нужно видеть человека 15 кадров, чтобы занять стол
REQUIRED_ABSENT_FRAMES = 75   # Нужно НЕ видеть человека 75 кадров, чтобы освободить

# Переменные состояния
events_log = []
last_vacated_time = None
table_occupied = False
last_event_was_vacated = True
approach_detected = False

# ИНИЦИАЛИЗАЦИЯ ВИДЕО
capture = cv2.VideoCapture(input_video_path)
fps = int(capture.get(cv2.CAP_PROP_FPS))
total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

ret, frame = capture.read()
if not ret:
    exit()
    

# Читаем первый кадр для выбора ROI
ret, frame = capture.read()
print("Выделите мышкой стол и нажмите ENTER. Для отмены - ESC.")
roi = cv2.selectROI("Select Table Zone", frame, fromCenter=False, showCrosshair=True)
tx, ty, tw, th = roi
cv2.destroyWindow("Select Table Zone")

reader = easyocr.Reader(['en'], gpu=False) # Инициализация OCR для распознавания времени на видео

# НАСТРОЙКА ЗАПИСИ ВЫХОДНОГО ВИДЕО
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
print(f"Видео будет сохранено в: {output_video_path}")

# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
def on_trackbar(val): # Функция для обработки перемещения ползунка
    capture.set(cv2.CAP_PROP_POS_FRAMES, val)

def get_timestamp(frame, x = 1446, y = 55, w = 474, h = 57): # Функция для извлечения времени из видео с помощью OCR. Координаты области с временем могут отличаться в зависимости от видео
    roi = frame[int(y) : int(y + h), int(x) : int(x + w)]
    
    if roi is None or roi.size == 0:
        return ""
        
    result = reader.readtext(roi, detail=0, allowlist='0123456789:.- ')
    
    return result[0] if result else ""

def is_person_at_table(box, tx, ty, tw, th, iou_threshold=0.15): # Функция для определения, находится ли человек в зоне стола. Используем IoU для оценки пересечения между рамкой человека и зоной стола. Если более 15% площади человека в зоне, считаем, что он у стола.
    x1, y1, x2, y2 = box.astype(int)
    person_box = [x1, y1, x2, y2]
    table_box = [tx, ty, tx+tw, ty+th]
    
    # Площадь пересечения
    inter_x1 = max(person_box[0], table_box[0])
    inter_y1 = max(person_box[1], table_box[1])
    inter_x2 = min(person_box[2], table_box[2])
    inter_y2 = min(person_box[3], table_box[3])
    
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return False
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    person_area = (x2 - x1) * (y2 - y1)
    
    return inter_area / person_area > iou_threshold   # если >15% человека в зоне

# ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО
cv2.namedWindow('Monitoring')
cv2.createTrackbar('Pos', 'Monitoring', 0, total_frames, on_trackbar)

while True:
    current_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
    cv2.setTrackbarPos('Pos', 'Monitoring', current_frame)
    
    ret, frame = capture.read()
    if not ret:
        break
        
    # Получаем результаты детекции для текущего кадра
    results = model(frame, classes=[0], verbose=False)[0]
    is_someone_at_table = False
    
    # Проходим по всем найденным людям и проверяем, находится ли кто-то в зоне стола
    for box in results.boxes.xyxy.cpu().numpy():
        if is_person_at_table(box, tx, ty, tw, th):
            is_someone_at_table = True
            x1,y1,x2,y2 = box.astype(int)
            cv2.rectangle(frame, (x1,y1),(x2, y2), (255,0,0), 2)
            
    if is_someone_at_table:
        presence_counter += 1
        absence_counter = 0 # Сбрасываем счетчик отсутствия, так как человек в зоне
    else:
        absence_counter += 1
        presence_counter = 0 # Сбрасываем счетчик присутствия
        
    # ЛОГИКА СОБЫТИЙ
    # 1. Подход к столу
    if not table_occupied and presence_counter >= 5 and not approach_detected and last_event_was_vacated: # Появился хотя бы на 5 кадров (~0.2 сек)
        raw_time = get_timestamp(frame).replace('.', ':')
        print(f"[{raw_time}] СОБЫТИЕ: Подход к столу")
        
        events_log.append({
            'timestamp': raw_time,
            'event': 'approach',
            'wait_after_previous': None
        })
        approach_detected = True
        last_event_was_vacated = False
    # 2. Занятие стола
    elif not table_occupied and presence_counter >= REQUIRED_PRESENT_FRAMES: # Человек виден уже 15 кадров (~0.8 сек)
        raw_time = get_timestamp(frame).replace('.',':')
        print(f"[{raw_time}] СОБЫТИЕ: Стол занят")

        current_dt = pd.to_datetime(raw_time, format='%d-%m-%Y %H:%M:%S', errors='coerce')
    
        wait_time = None
        if last_vacated_time is not None and pd.notnull(current_dt):
            wait_time = (current_dt - last_vacated_time).total_seconds()
    
        events_log.append({
            'timestamp': raw_time,
            'event': 'occupied',
            'wait_after_previous': wait_time
        })
        
        table_occupied = True
        approach_detected = False
        # 3. Освобождение стола
    elif table_occupied and absence_counter >= REQUIRED_ABSENT_FRAMES: # Человека нет уже 75 кадров (~3 сек)
        raw_time = get_timestamp(frame).replace('.',':')
        last_vacated_time = pd.to_datetime(raw_time, format='%d-%m-%Y %H:%M:%S', errors='coerce')
        events_log.append({
            'timestamp': raw_time,
            'event': 'vacated',
            'wait_after_previous': None
        })
        print(f"[{raw_time}] СОБЫТИЕ: Стол освободился")
        
        table_occupied = False
        last_event_was_vacated = True
        approach_detected = False

    # ВИЗУАЛИЗАЦИЯ
    # Рисуем статус стола и выбираем цвет рамки в зависимости от состояния: красный - занят, зеленый - свободен, серый - подход
    color = (0, 0, 255) if table_occupied else (0,255,0)
    status_text = "Занят" if table_occupied else "Пусто"
    if not table_occupied and presence_counter > 0:
        status_text = "Подход..." 
        color = (128,128,128)
    
    # Рисуем зону стола и статус
    cv2.rectangle(frame, (tx, ty), (tx + tw, ty + th), color, 2)
    cv2.putText(frame, status_text, (tx, ty - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
    
    # Сохраняем обработанный кадр в выходное видео и отображаем его
    out.write(frame)
    cv2.imshow("Monitoring", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        cv2.waitKey(-1)
        
# ФИНАЛЬНАЯ СТАТИСТИКА И СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
df = pd.DataFrame(events_log)
df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M:%S', errors='coerce')
df = df.sort_values('timestamp').reset_index(drop=True)

# Вычисляем время между событиями для подходов после освобождения
idle_times = []
for i in range(1, len(df)):
    prev_event = df.iloc[i-1]['event']
    curr_event = df.iloc[i]['event']
    if curr_event in ['approach', 'occupied'] and prev_event == 'vacated':
        delta = (df.iloc[i]['timestamp'] - df.iloc[i-1]['timestamp']).total_seconds()
        if pd.notnull(delta):
            idle_times.append(delta)

print("\n--- СТАТИСТИКА ---")
print(df)

if idle_times:
    mean_idle = sum(idle_times) / len(idle_times)
    print(f"\nСреднее время между уходом гостя и следующим подходом: {mean_idle:.1f} секунд")
    print(f"Всего оборотов (подходов после освобождения): {len(idle_times)}")
else:
    print("Нет данных для расчёта времени между гостями")

df.to_excel("table_stats.xlsx", index=False)

# ЗАВЕРШЕНИЕ РАБОТЫ
capture.release()
out.release()
cv2.destroyAllWindows()

print(f"\nВидео успешно сохранено в {output_video_path}")