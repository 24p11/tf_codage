import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class SlantedTriangularSchedule(LearningRateSchedule):
    """Slatend Trinagular Schedule from tensorflow"""

    def __init__(
        self, n_epochs, steps_per_epoch, max_learning_rate, ratio=32, cut_frac=0.1
    ):
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.cut_frac = cut_frac
        self.max_learning_rate = max_learning_rate
        self.ratio = ratio
        super().__init__()

    @tf.function
    def __call__(self, t):

        cut_frac = self.cut_frac
        ratio = self.ratio

        T = self.steps_per_epoch * self.n_epochs
        cut = tf.floor(T * cut_frac)
        p = t / cut if t < cut else 1 - (t - cut) / (cut * (1 / cut_frac - 1))
        learning_rate_step = self.max_learning_rate * (1 + p * (ratio - 1)) / ratio
        return tf.maximum(learning_rate_step, 0)

    def get_config(self):
        return {
            "n_epochs": self.n_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "cut_frac": self.cut_frac,
            "max_learning_rate": self.max_learning_rate,
            "ration": self.ratio,
        }
