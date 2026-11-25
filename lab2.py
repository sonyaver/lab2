import cv2
import os
import numpy as np
from datetime import datetime

BASE = "Lab2_Media"
INPUT_DIR = os.path.join(BASE, "data", "input")
OUTPUT_DIR = os.path.join(BASE, "data", "output")
LOG_FILE = os.path.join(BASE, "logs", "history.txt")


def log(filename, filter_name, status):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] Файл: {filename} | Фильтр: {filter_name} | Статус: {status}\n"
    with open(LOG_FILE, "a", encoding="UTF-8") as f:
        f.write(line)


def unsharp_mask(img, sigma=1.0, alpha=1.5):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)


def equalize_hist(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = cv2.equalizeHist(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)


def main():
    filename = input("Введите имя файла (image1.jpg / image2.png): ").strip()
    path = os.path.join(INPUT_DIR, filename)

    if not os.path.exists(path):
        print("Ошибка: Файл не найден")
        log(filename, "-", "Ошибка: файл не найден")
        return

    try:
        with open(path, "r+b") as f:
            data = f.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("Не удалось декодировать изображение")
    except PermissionError:
        print(f"Ошибка: Нет прав на чтение файла {filename}. "
              f"Обратитесь к администратору")
        log(filename, "-", "Ошибка доступа")
        return
    except IOError as e:
        print(f"Ошибка ввода-вывода: {e}")
        log(filename, "-", "Ошибка доступа")
        return

    result = unsharp_mask(img, sigma=1.0, alpha=1.5)
    result = equalize_hist(result)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "result.jpg")
    cv2.imwrite(out_path, result)

    print(f"Готово. Результат сохранён: {out_path}")
    log(filename, "Unsharp Masking + Equalization", "Успешно")


if __name__ == "__main__":
    main()
