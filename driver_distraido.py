from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


model = Sequential()
model.add(Convolution2D(nb_filter=32, nb_row=3,
                        nb_col=3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(nb_filter=64, nb_row=3, nb_col=3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# o model ate agora, tem como outputs mapas de funcionalidade 3d (height, width, features)
# converte nossos mapas de recursos 3D em vetores de recursos 1D
model.add(Flatten())
model.add(Dense(64))  # 64 neurons
model.add(Activation('relu'))
model.add(Dropout(0.5))  # Faz um drop de 50% dos neurons

# Camada de saida: Classifica os dez estados do motorista
model.add(Dense(10))
model.add(Activation('softmax'))

# Compila o model
model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', metrics=['accuracy'])

# Config de aumento para gerar training data
train_datagen = ImageDataGenerator(
    rescale=1.0/255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

# a validação de imagem esta escalada em 1/255, nenhum outro aumento para nossa validacao de dados
test_datagen = ImageDataGenerator(rescale=1.0/255)


# esse é o gerador que ira ler imagens en contradas em sub pastas de 'data/train',
# e indefinitivamente gera batch de dados de imagem aumentados!
train_generator = train_datagen.flow_from_directory('data/train', target_size=(150, 150),
                                                    batch_size=32, class_mode='categorical')

# Este é o gerador de dados de validação
validation_generator = test_datagen.flow_from_directory('data/validation', target_size=(150, 150),
                                                        batch_size=32, class_mode='categorical')

# trainando convolutional neural network
model.fit_generator(train_generator, samples_per_epoch=20924, nb_epoch=20,
                    validation_data=validation_generator, nb_val_samples=800)

# Salvando os weights
model.save_weights('driver_state_detection_small_CNN.h5')
