import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Leer los datos desde un archivo CSV
data = pd.read_csv('polynomial-regression.csv')


# Extraer los valores de X y Y excepto el último (leave-one-out)
X = data['araba_fiyat'].values[:-1]
y = data['araba_max_hiz'].values[:-1]

"""
# Extraer TODOS los valores de X y Y
X = data['araba_fiyat'].values
y = data['araba_max_hiz'].values
"""

# Calcular las sumatorias 

n = len(X) #Calcula lo muestreos
print("muestreos:", n)

sum_x = np.sum(X) #Calcula la sumatoria de X
print("sum_x:", sum_x)

sum_y = np.sum(y) #Calcula la sumatoria de y
print("sum_y:", sum_y)

sum_x_squared = np.sum(X**2) #Calcula la sumatoria de X^2
print("sum_x_squared:", sum_x_squared)

sum_x_cubed = np.sum(X**3) #Calcula la sumatoria de X^3
print("sum_x_cubed:", sum_x_cubed)

sum_x_fourth = np.sum(X**4) #Calcula la sumatoria de X^4
print("sum_x_fourth:", sum_x_fourth)

sum_x_fifth = np.sum(X**5) #Calcula la sumatoria de X^5
print("sum_x_fifth:", sum_x_fifth)

sum_x_sixth = np.sum(X**6) #Calcula la sumatoria de X^6
print("sum_x_sixth:", sum_x_sixth)

sum_x_y = np.sum(X * y) #Calcula la sumatoria de x*y
print("sum_x_y:", sum_x_y)

sum_x_squared_y = np.sum(X**2 * y)#Calcula la sumatoria de x^2*y
print("sum_x_squared_y:", sum_x_squared_y)

sum_x_cubed_y = np.sum(X**3 * y)#Calcula la sumatoria de x^3*y
print("sum_x_cubed_y:", sum_x_cubed_y)


print("\n")

#Imprime los sistemas de ecuaciones
print("Sistema de ecuaciones: n*b0 + b1*sum_x + b2*sum_x_squared = sum_y")
print("Sistema de ecuaciones: sum_x*b0 + b1*sum_x_squared + b2*sum_x_cubed = sum_x_y")
print("Sistema de ecuaciones: sum_x_squared*b0 + sum_x_cubed*b1 + b2*sum_x_fourth = sum_x_squared_y")

print("\n")

print("Sistema de ecuaciones:"+ str(n) + "*b0 + b1*" + str(sum_x) + " + b2*" + str(sum_x_squared) + " = " + str(sum_y))
print("Sistema de ecuaciones:"+ str(sum_x) + "*b0 + b1*" + str(sum_x_squared) + " + b2*" + str(sum_x_cubed) + " = " + str(sum_x_y))
print("Sistema de ecuaciones:"+ str(sum_x_squared) + "*b0 + " + str(sum_x_cubed) + "*b1 + b2*" + str(sum_x_fourth) + " = " + str(sum_x_squared_y))

print("\n")

# Crear una matriz aumentada para el sistema de ecuaciones
A = np.array([[n, sum_x, sum_x_squared, sum_y],
              [sum_x, sum_x_squared, sum_x_cubed, sum_x_y],
              [sum_x_squared, sum_x_cubed, sum_x_fourth, sum_x_squared_y]])

# Resolver el sistema utilizando Gauss-Jordan
XX = np.zeros(3)

for i in range(3):
    for j in range(i+1, 3):
        factor = A[j, i] / A[i, i]
        for k in range(4):
            A[j, k] -= factor * A[i, k]

for i in range(2, -1, -1):
    XX[i] = A[i, 3]
    for j in range(i+1, 3):
        XX[i] -= A[i, j] * XX[j]
    XX[i] /= A[i, i]

# Los valores de b0, b1 y b2 se encuentran en el vector X
b0, b1, b2 = XX

print("-----------------------Grado 2------------------------")

print(f'b0 = {b0}')
print(f'b1 = {b1}')
print(f'b2 = {b2}')

print("\n")

print("Modelo: y = " + str(b0) + " + " + str(b1) + "x + " + str(b2) + "x^2")

print("\n")

# Realizar predicciones
valor_a_predecir = data['araba_fiyat'].values[-1]  # Usa el ultimo valor de la columna X para predecir (leave-one-out)

resultado_b0 = b0
resultado_b1 = b1 * valor_a_predecir
resultado_b2 = valor_a_predecir**2 * b2

predicted_value = resultado_b0 + resultado_b1 + resultado_b2

print(f'Predicción para X={valor_a_predecir}: {predicted_value}')
print("\n")

#Grafica

# Gráfica de los datos reales
plt.scatter(X, y, label='Datos reales', color='blue')

# Gráfica del punto 'y real' que quieres destacar
ultimo_valor_x = data['araba_fiyat'].values[-1]
ultimo_valor_y = data['araba_max_hiz'].values[-1]
plt.scatter(ultimo_valor_x, ultimo_valor_y, label='Y real en ultimo valor', color='red')

# Gráfica de la regresión polinomial
x_range = np.linspace(0, 3000, 100)  # Rango de valores de x para la gráfica
y_pred = b0 + b1 * x_range + b2 * x_range**2
plt.plot(x_range, y_pred, label='Modelo de regresión polinomial simple', color='green')

# Gráfica del punto 'y predicho' que quieres destacar
plt.scatter(ultimo_valor_x, predicted_value, label='Y predecida en ultimo valor', color='purple')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Gráfica del Modelo de Regresión Polinomial grado 2')
plt.show()

#------------------------------------------------------------
#Implementacion de grado 3

# Ajuste de la regresión polinomial de grado 3
coefficients = np.polyfit(X, y, 3)

# Los coeficientes b0, b1, b2 y b3 son los elementos de 'coefficients'
b3, b2, b1, b0 = coefficients

print("-----------------------Grado 3------------------------")

print("b0 =", b0)
print("b1 =", b1)
print("b2 =", b2)
print("b3 =", b3)

print("\n")

print("Modelo: y = " + str(b0) + " + " + str(b1) + "x + " + str(b2) + "x^2 + " + str(b3) + "x^3")

print("\n")

# Realizar predicciones
valor_a_predecir = data['araba_fiyat'].values[-1]  # Usa el ultimo valor de la columna X para predecir (leave-one-out)

resultado_b0 = b0
resultado_b1 = b1 * valor_a_predecir
resultado_b2 = valor_a_predecir**2 * b2
resultado_b3 = valor_a_predecir**3 * b3

predicted_value = resultado_b0 + resultado_b1 + resultado_b2 + resultado_b3

print(f'Predicción para X={valor_a_predecir}: {predicted_value}')
print("\n")

#Grafica

# Gráfica de los datos reales
plt.scatter(X, y, label='Datos reales', color='blue')

# Gráfica del punto 'y real' que quieres destacar
ultimo_valor_x = data['araba_fiyat'].values[-1]
ultimo_valor_y = data['araba_max_hiz'].values[-1]
plt.scatter(ultimo_valor_x, ultimo_valor_y, label='Y real en ultimo valor', color='red')

# Gráfica de la regresión polinomial
x_range = np.linspace(0, 3000, 100)  # Rango de valores de x para la gráfica
y_pred = b0 + b1 * x_range + b2 * x_range**2 + b3 * x_range**3
plt.plot(x_range, y_pred, label='Modelo de regresión polinomial simple', color='green')

# Gráfica del punto 'y predicho' que quieres destacar
plt.scatter(ultimo_valor_x, predicted_value, label='Y predecida en ultimo valor', color='purple')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Gráfica del Modelo de Regresión Polinomial grado 3')
plt.show()


#------------------------------------------------------------
#Implementacion de grado 4

# Ajuste de la regresión polinomial de grado 4
coefficients = np.polyfit(X, y, 4)

# Los coeficientes b0, b1, b2 y b3 son los elementos de 'coefficients'
b4, b3, b2, b1, b0 = coefficients

print("-----------------------Grado 4------------------------")

print("b0 =", b0)
print("b1 =", b1)
print("b2 =", b2)
print("b3 =", b3)
print("b4 =", b4)

print("\n")

print("Modelo: y = " + str(b0) + " + " + str(b1) + "x + " + str(b2) + "x^2 + " + str(b3) + "x^3" + str(b4) + "x^4")

print("\n")

# Realizar predicciones
valor_a_predecir = data['araba_fiyat'].values[-1]  # Usa el ultimo valor de la columna X para predecir (leave-one-out)

resultado_b0 = b0
resultado_b1 = b1 * valor_a_predecir
resultado_b2 = valor_a_predecir**2 * b2
resultado_b3 = valor_a_predecir**3 * b3
resultado_b4 = valor_a_predecir**4 * b4

predicted_value = resultado_b0 + resultado_b1 + resultado_b2 + resultado_b3 + resultado_b4


print(f'Predicción para X={valor_a_predecir}: {predicted_value}')
print("\n")

#Grafica

# Gráfica de los datos reales
plt.scatter(X, y, label='Datos reales', color='blue')

# Gráfica del punto 'y real' que quieres destacar
ultimo_valor_x = data['araba_fiyat'].values[-1]
ultimo_valor_y = data['araba_max_hiz'].values[-1]
plt.scatter(ultimo_valor_x, ultimo_valor_y, label='Y real en ultimo valor', color='red')

# Gráfica de la regresión polinomial
x_range = np.linspace(0, 3000, 100)  # Rango de valores de x para la gráfica
y_pred = b0 + b1 * x_range + b2 * x_range**2 + b3 * x_range**3 + b4 * x_range**4
plt.plot(x_range, y_pred, label='Modelo de regresión polinomial simple', color='green')

# Gráfica del punto 'y predicho' que quieres destacar
plt.scatter(ultimo_valor_x, predicted_value, label='Y predecida en ultimo valor', color='purple')

plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Gráfica del Modelo de Regresión Polinomial grado 4')
plt.show()

#------------------------------------------------------------
#Implementacion de grado 5

# Ajuste de la regresión polinomial de grado 5
coefficients = np.polyfit(X, y, 5)

# Los coeficientes b0, b1, b2 y b3 son los elementos de 'coefficients'
b5, b4, b3, b2, b1, b0 = coefficients

print("-----------------------Grado 5------------------------")

print("b0 =", b0)
print("b1 =", b1)
print("b2 =", b2)
print("b3 =", b3)
print("b4 =", b4)
print("b5 =", b5)

print("\n")

print("Modelo: y = " + str(b0) + " + " + str(b1) + "x + " + str(b2) + "x^2 + " + str(b3) + "x^3" + str(b4) + "x^4" + str(b5) + "x^5")

print("\n")
