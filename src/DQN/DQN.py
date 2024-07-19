from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import Adam

def create_dqn(learn_rate, input_dims, n_actions, conv_units, dense_units):
    model = Sequential(
                [
                    Conv2D(conv_units, (3,3), activation='relu', padding='same', input_shape=input_dims),
                    Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                    Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                    Conv2D(conv_units, (3,3), activation='relu', padding='same'),
                    Flatten(),
                    Dense(dense_units, activation='relu'),
                    Dense(dense_units, activation='relu'),
                    Dense(n_actions, activation='linear')
                ]
            )

    model.compile(optimizer=Adam(learning_rate=learn_rate, epsilon=1e-4), loss='mse')

    return model

if __name__ == "__main__":
    model = create_dqn(learn_rate=0.001, input_dims=(9, 9, 1), n_actions=81, conv_units=64, dense_units=512)
    model.summary()