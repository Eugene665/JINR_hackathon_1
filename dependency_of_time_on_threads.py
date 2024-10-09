import numpy as np
import matplotlib.pyplot as plt
import joblib

# Автоматическое получение количества доступных потоков
num_threads = joblib.cpu_count()

# Генерация данных для количества потоков от 1 до num_threads
threads = np.arange(1, num_threads + 1)

# Либо же статическое количества потоков
#threads = np.arange(1, 21)  # Количество потоков от 1 до 20

# Время выполнения, уменьшающееся с увеличением количества потоков(в теории)
execution_time = 100 / threads + np.random.normal(0, 2, size=len(threads))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(threads, execution_time, marker='o', linestyle='-', color='blue')

plt.title('Зависимость времени выполнения от количества потоков')
plt.xlabel('Количество потоков')
plt.ylabel('Время выполнения (секунды)')
plt.xticks(threads)
plt.grid()
plt.show()
