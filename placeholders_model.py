import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(8, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(16, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = np.random.rand(30,64,64,1)
y = np.eye(3)[np.random.randint(0,3,30)]

model.fit(X, y, epochs=2)
model.save("battery_model.h5")
print("Model saved as battery_model.h5")
