import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data1 = {
    "N": [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102],
    "D": [3, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
    "D_avg": [1.933, 2.303, 2.359, 2.496, 2.581, 2.734, 2.750, 2.855, 2.908, 2.977, 3.015, 3.064, 3.092, 3.128, 3.150, 3.178, 3.195],
    "S": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    "C": [5, 16, 31, 46, 63, 77, 93, 107, 123, 137, 153, 167, 183, 197, 213, 227, 243],
    "T": [1.288, 1.151, 0.943, 0.832, 0.737, 0.683, 0.611, 0.571, 0.528, 0.496, 0.463, 0.437, 0.412, 0.391, 0.370, 0.353, 0.336]
}

data2 = {
    "N": [7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105],
    "D": [2, 3, 3, 4, 4, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8],
    "D_avg": [1.619, 1.945, 2.028, 2.248, 2.457, 2.627, 2.839, 2.992, 3.195, 3.335, 3.525, 3.659, 3.838, 3.958, 4.134],
    "S": [6, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
    "C": [8, 25, 51, 70, 88, 108, 125, 144, 161, 180, 197, 216, 233, 252, 269],
    "T": [0.539, 0.486, 0.450, 0.499, 0.546, 0.583, 0.630, 0.664, 0.710, 0.741, 0.783, 0.813, 0.852, 0.879, 0.918]
}

data3 = {
    "N": [9, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99, 108],
    "D": [3, 7, 7, 6, 8, 8, 9, 8, 8, 10, 10, 10],
    "D_avg": [1.805, 2.856, 2.982, 2.936, 3.413, 3.498, 3.781, 3.711, 3.801, 4.299, 4.245, 4.356],
    "S": [4, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    "C": [13, 27, 45, 66, 82, 103, 121, 143, 164, 180, 202, 223],
    "T": [0.902, 1.428, 1.193, 0.978, 1.137, 1.166, 1.260, 1.237, 1.267, 1.433, 1.415, 1.452]
}

df1 = pd.DataFrame(data1)
df2 = pd.DataFrame(data2)
df3 = pd.DataFrame(data3)

plt.figure(figsize=(16, 10))

plt.subplot(2, 3, 1)
plt.plot(df1["N"], df1["D"], label="Зірка")
plt.plot(df2["N"], df2["D"], label="Кільце")
plt.plot(df3["N"], df3["D"], label="Решітка")
plt.title("Залежність D від N")
plt.xlabel("N (Кількість процесорів)")
plt.ylabel("D")
plt.legend()
plt.grid()

plt.subplot(2, 3, 2)
plt.plot(df1["N"], df1["D_avg"], label="Зірка")
plt.plot(df2["N"], df2["D_avg"], label="Кільце")
plt.plot(df3["N"], df3["D_avg"], label="Решітка")
plt.title("Залежність $\\bar{D}$ від N")
plt.xlabel("N (Кількість процесорів)")
plt.ylabel("$\\bar{D}$")
plt.legend()
plt.grid()

plt.subplot(2, 3, 3)
plt.plot(df1["N"], df1["S"], label="Зірка")
plt.plot(df2["N"], df2["S"], label="Кільце")
plt.plot(df3["N"], df3["S"], label="Решітка")
plt.title("Залежність S від N")
plt.xlabel("N (Кількість процесорів)")
plt.ylabel("S")
plt.legend()
plt.grid()

plt.subplot(2, 3, 4)
plt.plot(df1["N"], df1["C"], label="Зірка")
plt.plot(df2["N"], df2["C"], label="Кільце")
plt.plot(df3["N"], df3["C"], label="Решітка")
plt.title("Залежність C від N")
plt.xlabel("N (Кількість процесорів)")
plt.ylabel("C")
plt.legend()
plt.grid()

plt.subplot(2, 3, 5)
plt.plot(df1["N"], df1["T"], label="Зірка")
plt.plot(df2["N"], df2["T"], label="Кільце")
plt.plot(df3["N"], df3["T"], label="Решітка")
plt.title("Залежність T від N")
plt.xlabel("N (Кількість процесорів)")
plt.ylabel("T")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Data for the bar chart

data1 = [5.729, 6.474, 7.505]  # First topology

data2 = [6.418, 6.167, 6.443]   # Second topology

data3 = [6.142, 5.87, 6.448]  # Third topology



# Labels for x-axis

categories = ['Зона 1', 'Зона 2', 'Зона 3']



# Define the positions for bars

x1 = np.arange(len(categories))

x2 = x1 + len(categories) + 1  # Add gap for the second topology

x3 = x2 + len(categories) + 1  # Add gap for the third topology



# Plotting the bar chart

plt.figure(figsize=(10, 6))



plt.bar(x1, data1, width=0.8, label='Зірка', color='blue')

plt.bar(x2, data2, width=0.8, label='Кільце', color='green')

plt.bar(x3, data3, width=0.8, label='Решітка', color='orange')



# Customizing the chart

plt.title('Порівняння суми нормалізованих результатів')


plt.ylabel('Значення')

plt.xticks(np.concatenate([x1, x2, x3]), categories * 3, rotation=45)

plt.legend()

plt.grid(axis='y', linestyle='--', alpha=0.7)



# Show the chart

plt.tight_layout()

plt.show()
