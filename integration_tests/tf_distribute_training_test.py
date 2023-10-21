import numpy as np
import tensorflow as tf

import keras_core as keras
from keras_core import layers
from keras_core import losses
from keras_core import metrics
from keras_core import models
from keras_core import optimizers
from keras_core.callbacks import LearningRateScheduler


def simple_scheduler():
    """Create a LearningRateScheduler with step decay schedule.

    Args:
        initial_lr (float): Initial learning rate
        decay_factor (float): Learning rate decay factor
        step_size (int): Interval size on decay in epochs

    Returns:
        keras.callbacks.LearningRateScheduler
    """

    def _schedule(epoch):
        return 0.1

    return LearningRateScheduler(_schedule)

def test_model_fit():
    cpus = tf.config.list_physical_devices("CPU")
    tf.config.set_logical_device_configuration(
        cpus[0],
        [
            tf.config.LogicalDeviceConfiguration(),
            tf.config.LogicalDeviceConfiguration(),
        ],
    )

    keras.utils.set_random_seed(1337)

    strategy = tf.distribute.MirroredStrategy(["CPU:0", "CPU:1"])
    with strategy.scope():
        inputs = layers.Input((100,), batch_size=32)
        x = layers.Dense(256, activation="relu")(inputs)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        outputs = layers.Dense(16)(x)
        model = models.Model(inputs, outputs)

    model.summary()

    x = np.random.random((5000, 100))
    y = np.random.random((5000, 16))
    batch_size = 32
    epochs = 2
    callbacks = [simple_scheduler()]

    # Fit from numpy arrays:
    with strategy.scope():
        model.compile(
            optimizer=optimizers.SGD(learning_rate=0.001, momentum=0.01),
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            # TODO(scottzhu): Find out where is the variable
            #  that is not created eagerly and break the usage of XLA.
            jit_compile=False
        )
        history = model.fit(
            x, y, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=callbacks
        )

    print("History:")
    print(history.history)

    # Fit again from distributed dataset:
    with strategy.scope():
        dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)
        dataset = strategy.experimental_distribute_dataset(dataset)
        history = model.fit(
            dataset, epochs=epochs, callbacks=callbacks
        )

    print("History:")
    print(history.history)


if __name__ == "__main__":
    test_model_fit()
