import numpy as np
import matplotlib.pyplot as plt

# 1. Создаем массив значений G с равномерно
# распределёнными 200-ми точками от 0.1 до 10
G = np.linspace(0.1, 10, 200)

# 2. Для примера вводим случайные значения
# средних mx, my, mz длиной в G(200 значений)
arr_mean_mx = np.sin(G) + 0.5 * np.random.normal(size=len(G))
arr_mean_my = np.cos(G) + 0.5 * np.random.normal(size=len(G))
arr_mean_mz = 0.5 * G + 0.5 * np.random.normal(size=len(G))

# 3. Построение графика
plt.figure(figsize=(10, 6))

plt.plot(G, arr_mean_mx, label='Mean mx', color='blue')
plt.plot(G, arr_mean_my, label='Mean my', color='green')
plt.plot(G, arr_mean_mz, label='Mean mz', color='red')

# Настройки графика
plt.title('Зависимость средних значений от G')
plt.xlabel('G')
plt.ylabel('Средние значения')
plt.legend()
plt.grid()
plt.show()