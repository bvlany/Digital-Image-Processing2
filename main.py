import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print("Напишите текст для горизонтального вектора")
epoch = input()
print("Напишите текст для вертикального вектора")
acc = input()
print("Напишите целевую продуктивность")
etrics = input()
print("Напишите вашу продуктивность")
a = input()
# загрузка данных MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# нормализация данных
x_train = x_train / 255.0
x_test = x_test / 255.0

# определение модели нейросети
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# компиляция модели
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# обучение модели
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# оценка модели на тестовых данных
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

# графики обучения модели
plt.plot(history.history['accuracy'], label=etrics)
plt.plot(history.history['val_accuracy'], label=a)
plt.xlabel(epoch)
plt.ylabel(acc)
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

