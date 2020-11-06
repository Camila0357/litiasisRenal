# litiasisRenal
Proyecto de Grado II - Ingeniería de Sistemas UNAB

El presente proyecto se muestra como una solución inicial para la detección de litiasis renal por ecografía, haciendo uso del framework de TensorFlow y la arquitectura de ResNet50, con una capa densa de 256 y un Dropout de 0.5

Para la selección del mejor modelo, se partió de una arquitectura pre entrenada con un Total de 23,587,712 de parámetros, de los cuales 23,534,592 son entrenables y 53,120 no son entrenables, la arquitectura con mejor desempeño fue una basada en ResNet50, previamente entrenada con las imágenes de ImageNET y posteriormente sometida al proceso de Transfer Learning. 

Así mismo se añadieron manualmente varias capas: una capa de vectorización Flatten, seguida de una capa densa con 256 neuronas de capas totalmente conectadas, luego de una capa de Dropout al 50% para la regularización del modelo y finalmente la capa de salida de 2 neuronas (Una por cada posible estado de salida, “con litiasis” o “sin litiasis”) y una función de activación softmax para convertir la salida probabilística en un valor binario 1 / 0. 

El modelo se entrena con la versión más reciente a la fecha de Tensorflow 2.3.0, haciendo uso de la herramienta de ETL automatizada de imágenes (ImageDataGenerator) que facilita automáticamente las operaciones de carga, transformación y ajuste de las imágenes para ser enviadas en bloque al método fit_generator() propio de los modelos construidos bajo este framework.

Se definen adicionalmente dos funciones de Callback para generar automáticamente “checkpoints” de los modelos con buen desempeño, y para detener el entrenamiento dado el caso de que por cinco épocas consecutivas no haya ningún progreso en el aprendizaje del modelo.


