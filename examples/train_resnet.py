# deepSKAN
# This shows how to train a resNet-Like CNN for kinetic model predicition
# The training data is generated on the spot by the onlineDataGenerator class

from multiprocessing import freeze_support

def run():
    import keras
    import tensorflow as tf
    from keras.models import Sequential, Model
    from keras.layers import Dense, Dropout, Flatten, GaussianNoise 
    from keras.layers import AveragePooling2D, BatchNormalization
    from keras.layers import Conv2D, MaxPooling2D, LSTM, CuDNNLSTM
    from keras.layers import Reshape, Conv1D, Reshape
    from keras.layers import Input, ReLU, Activation 
    from keras.layers import SeparableConv2D, GlobalAveragePooling2D
    from keras.layers.merge import add
    from keras import layers
    from keras.callbacks import EarlyStopping, ModelCheckpoint 
    from keras import backend as K

    from deepGTA import onlineDataGenerator
    from deepGTA.utils import top_10_acc

    batch_size = 16
    epoch_size = 2**16
    val_split = 8
    learning_rate = 0.0001
    epochs = 64
    activation = 'relu'

    # Init generators
    training_generator = onlineDataGenerator(batch_size, epoch_size)
    validation_generator = onlineDataGenerator(batch_size, int(epoch_size/val_split))

    # A residual block adds skip connections between its input and output
    def residual_block(y, nb_channels, kernel_size, num_operations, 
                       project_identity=True):
        shortcut = y

        for _ in range(num_operations-1):
            y = Conv2D(nb_channels, kernel_size, padding='same')(y)
            y = Activation(activation)(y)

        y = Conv2D(nb_channels, kernel_size, padding='same')(y)
        
        if project_identity: 
            shortcut = Conv2D(nb_channels, kernel_size=(1, 1), padding='same')(shortcut)

        y = add([shortcut, y])
        y = Activation(activation)(y)

        return y

    # Define the actual model architecture

    # input layers
    input_layer = Input((256, 64, 1), name='input_noise_layer')
    noise = GaussianNoise(0.01)(input_layer)

    # Residual blocks along time dimension
    res = residual_block(noise,64, (15,1), 4)
    mp = MaxPooling2D((2,1))(res)

    res = residual_block(mp, 128, (11,1), 4)
    mp = MaxPooling2D((2,1))(res)

    # Residual blocks along both dimensions
    res = residual_block(mp, 128, (9,3), 4, project_identity=True)
    mp = MaxPooling2D((2,2))(res)

    res = residual_block(mp, 128, (9,3), 4, project_identity=False)
    mp = MaxPooling2D((2,2))(res)

    res = residual_block(mp, 256, (5,5), 4)
    mp = MaxPooling2D((2,2))(res)

    res = residual_block(mp, 256, (3,3), 4, project_identity=False)
    mp = MaxPooling2D((2,2))(res)
    res = residual_block(mp, 128, (3,3), 2)

    x = Flatten()(res)

    # Dense layers at the end
    x = Dense(1024, activation=activation)(x)

    x = Dense(512, activation=activation)(x)

    x = Dense(512, activation=activation)(x)

    x = Dense(256, activation=activation)(x)

    output_layer = Dense(103, activation='softmax', name='ouput_dense_layer')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    keras.metrics.top_10_acc = top_10_acc


    # compile model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=learning_rate, beta_1=0.9, 
                  beta_2=0.999, epsilon=1e-3, amsgrad=False),
                  metrics=['accuracy', keras.metrics.top_k_categorical_accuracy, 
                           top_10_acc]
                  )
    # add callbacks
    stop_here_please = EarlyStopping(patience=10, restore_best_weights=True)
    lr_plateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.1, patience=3,
                                  verbose=1, mode='auto', min_delta=0.01, 
                                  cooldown=1)
    print(model.summary())

    # Start training
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,
                        max_queue_size=128,
                        workers=4,
                        validation_steps=int(epoch_size/val_split/batch_size),
                        epochs=epochs,
                        callbacks=[stop_here_please, lr_plateau])

    model.save('resnet_10')


if __name__ == '__main__':
    freeze_support()
    run()