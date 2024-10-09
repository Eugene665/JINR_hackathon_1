def find_stabilization(arr, required_length, tolerance, outlier_limit):
    current_length = 0
    stabilization_start = 0
    outlier_count = 0

    # Проходим по списку значений, сравнивая значения через один элемент
    for i in range(len(arr) - 2):  # -2, чтобы избежать выхода за границы
        # Сравниваем числа через одно ввиду чередуемости
        # Пример массива
        # 1. 0.85
        # 2. 0.80
        # 3. 0.85
        # 4. 0.80
        # 5. 0.85
        # 6. 0.80
        # Нужно сравнить первое и третье,  второе и четвёртое число и т.д.
        # Если при сравнение модуль разности < погрешности, то увеличиваем актуальную длину
        if abs(arr[i] - arr[i + 2]) <= tolerance:
            # Если последовательность новая, то запоминаем её индекс для аналитики
            if current_length == 0:
                stabilization_start = i
            current_length += 1
        else:
            # Если значение выходит за пределы tolerance, увеличиваем счетчик выбросов
            outlier_count += 1

            # Если количество выбросов превышает допустимый предел,
            # сбрасываем последовательность и счетчик выбросов
            if outlier_count > outlier_limit:
                current_length = 0
                outlier_count = 0
                # Пропускаем итерацию
                continue

            # Если последовательность прерывается, выводим информацию о интервалах чередования
            if current_length >= required_length:
                print(f"Стабилизация начинается с индекса {stabilization_start} и продолжается {current_length} пар")
            # Сбрасываем счетчик текущей длины
            current_length = 0

    # Проверка в конце списка
    if current_length >= required_length:
        print(f"Стабилизация начинается с индекса {stabilization_start} и продолжается {current_length} пар")

# В тестовом варианте считываем данные из файла txt
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Преобразуем строки в список чисел
arr_mx = [float(line.strip().split('=')[-1]) for line in lines if 'Maximum' in line]

# Устанавливаем требуемую длину стабилизации, погрешность при
# сравнении и колличество выбросов для длины
required_length = 100
tolerance = 0.02
outlier_limit = 5

# Ищем момент стабилизации
find_stabilization(arr_mx, required_length, tolerance, outlier_limit)
