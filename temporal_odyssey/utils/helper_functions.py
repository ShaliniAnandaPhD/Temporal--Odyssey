import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import logging
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import kerastuner as kt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

# Function for model checkpointing and early stopping
def setup_callbacks(checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.h5')
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True)
    return [early_stopping, model_checkpoint]

# Function to load the best model
def load_best_model(checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.h5')
    if os.path.exists(checkpoint_path):
        model = load_model(checkpoint_path)
        logger.info(f"Loaded best model from {checkpoint_path}")
        return model
    else:
        logger.error(f"No model found at {checkpoint_path}")
        return None

# Function for flexible hyperparameter tuning
def hyperparameter_tuning(model_builder, param_grid, x_train, y_train, x_val, y_val, max_trials=10):
    class CustomTuner(kt.Tuner):
        def run_trial(self, trial, *args, **kwargs):
            model = model_builder(trial)
            callbacks = setup_callbacks('checkpoints')
            return model.fit(
                *args,
                **kwargs,
                callbacks=callbacks,
            )
    
    tuner = CustomTuner(
        oracle=kt.oracles.RandomSearch(
            objective='val_loss',
            max_trials=max_trials
        ),
        hypermodel=model_builder,
        directory='tuner',
        project_name='hyperparam_tuning'
    )
    tuner.search(x_train, y_train, validation_data=(x_val, y_val))
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_model, best_hyperparameters

# Example usage of flexible logging
def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

# Example model builder function for hyperparameter tuning
def build_model(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])), loss='mse')
    return model

# Example usage
if __name__ == "__main__":
    # Set up logging
    setup_logging('logs')

    # Load data
    x_train, y_train, x_val, y_val = np.random.rand(100, 10), np.random.rand(100, 1), np.random.rand(20, 10), np.random.rand(20, 1)

    # Hyperparameter tuning
    best_model, best_hyperparameters = hyperparameter_tuning(build_model, {}, x_train, y_train, x_val, y_val)

    # Load best model
    best_model = load_best_model('checkpoints')

    # Train the model with early stopping and model checkpointing
    callbacks = setup_callbacks('checkpoints')
    best_model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, callbacks=callbacks)

    # Evaluate the model
    loss = best_model.evaluate(x_val, y_val)
    logger.info(f"Validation loss: {loss}")

# Function for preprocessing state
def preprocess_state(state):
    return np.reshape(state, [1, state.shape[0]])
