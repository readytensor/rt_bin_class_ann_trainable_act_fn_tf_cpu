import os
import warnings
from typing import Callable, Dict, Optional, Union, Tuple, List

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.exceptions import NotFittedError
from tensorflow.keras.callbacks import Callback, EarlyStopping, LambdaCallback
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from logger import get_logger

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}
warnings.filterwarnings("ignore")


MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"
HISTORY_FNAME = "history.json"
COST_THRESHOLD = float("inf")


logger = get_logger(task_name="tf_model_training")

# Check TensorFlow Version
logger.info(f"TensorFlow Version: {tf.__version__}")

# Check for GPU availability
gpu_avai = (
    "GPU available (YES)"
    if tf.config.list_physical_devices("GPU")
    else "GPU not available"
)
logger.info(gpu_avai)


def create_logger(log_period: int, log_type: str = "epoch") -> Callable:
    """
    Create a logging function to log information every log_period epochs or batches.

    This function creates and returns another function:
        `log_function(log_count, logs)`
    which checks if the current log_count number (either epoch or batch number)
    is a multiple of the specified log_period. If it is, it logs the log_count number
    and the logs information.

    Args:
        log_period (int): The period at which to log information. For example, if
                    log_period is 10, the logging will happen at every 10th epoch
                    or batch (e.g., 0th, 10th, 20th, etc.)
        log_type (str): A string that is either 'epoch' or 'batch' specifying the
                    type of logging.
                    Defaults to 'epoch'.

    Returns:
        Callable: The log_function function that logs every log_period epochs
                    or batches.
    """

    def log_function(log_count: int, logs: Dict) -> None:
        logs_str = ""
        for k, v in logs.items():
            logs_str += f"{k}: {np.round(v, 4)}  "
        if log_count % log_period == 0:
            logger.info(f"{log_type.capitalize()}: {log_count}, Metrics: {logs_str}")

    return log_function


class InfCostStopCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        loss_val = logs.get("loss")
        if loss_val == COST_THRESHOLD or tf.math.is_nan(loss_val):
            logger.warning("Cost is inf, so stopping training!!")
            self.model.stop_training = True


def get_init_values(shape: Union[Tuple[int, ...], List[int]]) -> np.ndarray:
    """
    Initializes values with given shape using random normal distribution.

    Args:
        shape (Union[Tuple[int, ...], List[int]]): The shape of the values to be
            initialized. This can be a tuple or list of integers.

    Returns:
        np.ndarray: Initialized values reshaped into the given shape.

    Examples:
        >>> shape = (2, 3)
        >>> vals = get_init_values(shape)
        >>> print(vals.shape)
        (2, 3)

    Notes:
        The values are drawn from a standard normal distribution (mean=0, stdev=1).
    """
    dim = np.prod(shape)
    vals = np.random.randn(dim)
    return vals.reshape(shape)


class TrainableActivationLayer(Layer):
    """
    Custom layer for trainable activation function.

    Attributes:
        num_cps (int): Number of change points for the activation function.
        slopes (tf.Variable): Variable for slopes of the activation function.
        intercepts (tf.Variable): Variable for intercepts of the activation function.
        locations (tf.Variable): Variable for locations of the change points.
        lambda_ (tf.Variable): Variable for the lambda parameter of the activation
            function.
    """

    def __init__(self, num_cps: int):
        super(TrainableActivationLayer, self).__init__()
        self.num_cps = num_cps

    def build(self, input_shape: Tuple[int, ...]):
        """
        Create the layer's variables.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input.
        """
        D = input_shape[-1]
        shape = (1, D, self.num_cps)
        self.locations = tf.Variable(
            initial_value=get_init_values(shape), dtype=tf.float32
        )  # shape => (1, D, num_cps)
        self.location_values = tf.Variable(
            initial_value=get_init_values(shape), dtype=tf.float32
        )  # shape => (1, D, num_cps)
        shape = (1, D, 1)
        self.lambda_ = tf.Variable(
            initial_value=get_init_values(shape), dtype=tf.float32
        )  # shape => (1, D, 1)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Compute the output of the layer.

        Args:
            inputs (tf.Tensor): Input tensor.
        
        Returns:
            tf.Tensor: Output tensor after applying the trainable activation function.
        """
        inputs = tf.expand_dims(
            inputs, axis=2, name="expand_dims"
        )  # shape goes from (NxD) => (N, D, 1)
        sq_diff = tf.math.square(
            inputs - self.locations, name="sq_diff"
        )  # shape = (N, D, num_cps)
        exp = tf.math.exp(
            -self.lambda_ * sq_diff, name="exp"
        )  # shape = (N, D, num_cps)
        probs = tf.nn.softmax(exp, name="softmax")  # shape = (N, D, num_cps)

        loc_vals_x_probs = self.location_values * probs # shape = (N, D, num_cps)

        output_ = tf.math.reduce_sum(
            loc_vals_x_probs, axis=-1, name="output"
        )  # shape = (N, D)
        output_ = tf.math.tanh(output_)
        return output_



class Classifier:
    """
    A wrapper class for the ANN with Training Activation Function Binary classifier
    in Tensorflow.
    """

    model_name = "ANN with Training Activation Function Binary classifier - TF-CPU"

    def __init__(
        self,
        D: Optional[int] = None,
        lr: Optional[float] = 1e-4,
        num_cps: Optional[int] = 2,
        **kwargs,
    ):
        """Construct a new binary classifier.

        Args:
            D (int, optional): Size of the input layer.
                Defaults to None (set in `fit`).
            lr (int, optional): Learning rate for optimizer.
                Defaults to 1e-4.
            num_cps (int, optional): Number of change points in activitation
                function.
                Range: 1 <=num_cps <= 3
                Defaults to 2.
        """
        self.D = D
        self.lr = lr
        self.num_cps = num_cps
        self._log_period = 10  # logging per 10 epochs
        # defer building model until fit because we need to know
        # dimensionality of data (D) to define the size of
        # input layer
        self.model = None

    def build_model(self):
        input_ = Input(self.D)
        x = input_
        x = Dense(max(100, self.D*4))(x)
        x = TrainableActivationLayer(num_cps=self.num_cps)(x)
        x = Dense(max(20, self.D*0.5))(x)
        x = TrainableActivationLayer(num_cps=self.num_cps)(x)
        x = Dense(1, activation="sigmoid")(x)
        output_ = x
        model = Model(input_, output_, name="binary_classifier_ann_train_act_fn")
        # model.summary()
        model.compile(
            loss=BinaryCrossentropy(),
            optimizer=Adam(learning_rate=self.lr),
            metrics=["accuracy"],
        )
        return model

    def fit(
        self,
        train_inputs: pd.DataFrame,
        train_targets: pd.Series,
        batch_size=100,
        epochs=750,
    ) -> None:
        """Fit the classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        # get data dimensionality and build network
        self.D = train_inputs.shape[1]
        self.model = self.build_model()

        # set seed for reproducibility
        tf.random.set_seed(0)

        # use 15% validation split if at least 300 samples in training data
        if train_inputs.shape[0] < 300:
            loss_to_monitor = "loss"
            validation_split = None
        else:
            loss_to_monitor = "val_loss"
            validation_split = 0.15

        early_stop_callback = EarlyStopping(
            monitor=loss_to_monitor, min_delta=1e-3, patience=30
        )
        infcost_stop_callback = InfCostStopCallback()
        logger_callback = LambdaCallback(
            on_epoch_end=create_logger(self._log_period, "epoch")
        )

        self.model.fit(
            x=train_inputs,
            y=train_targets,
            batch_size=batch_size,
            validation_split=validation_split,
            epochs=epochs,
            shuffle=True,
            verbose=False,
            callbacks=[
                early_stop_callback,
                infcost_stop_callback,
                logger_callback,
            ],
        )

    def _predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        return self.model.predict(inputs, verbose=False)

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        class1_probs = self._predict(inputs).reshape(-1, 1)
        predicted_labels = (class1_probs >= 0.5).astype(int)
        return np.squeeze(predicted_labels)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """
        class1_probs = self._predict(inputs).reshape(-1, 1)
        class0_probs = 1.0 - class1_probs
        probs = np.hstack((class0_probs, class1_probs))
        return probs

    def summary(self):
        """Return model summary of the Tensorflow model"""
        self.model.summary()

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the binary classifier.
        """
        if self.model is not None:
            # returns list containing loss value and metric value
            # index at 1 which contains accuracy
            return self.model.evaluate(test_inputs, test_targets, verbose=0)[1]
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the binary classifier to disk.

        Args:
            model_dir_path (str): The dir path to which to save the model.
        """
        if self.model is None:
            raise NotFittedError("Model is not fitted yet.")
        model_params = {
            "D": self.D,
            "lr": self.lr,
            "num_cps": self.num_cps,
        }
        joblib.dump(model_params, os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        self.model.save_weights(os.path.join(model_dir_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded binary classifier.
        """
        if not os.path.exists(model_dir_path):
            raise FileNotFoundError(f"Model dir {model_dir_path} does not exist.")
        model_params = joblib.load(os.path.join(model_dir_path, MODEL_PARAMS_FNAME))
        classifier_model = cls(**model_params)
        classifier_model.model = classifier_model.build_model()
        classifier_model.model.load_weights(
            os.path.join(model_dir_path, MODEL_WTS_FNAME)
        ).expect_partial()
        return classifier_model

    def __str__(self):
        return (
            f"Model name: {self.model_name}\n"
            f"D: {self.D}\n"
            f"l1_reg: {self.l1_reg}\n"
            f"l2_reg: {self.l2_reg}\n"
            f"lr: {self.lr})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_X (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, model_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        model_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model.save(model_dir_path)


def load_predictor_model(model_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        model_dir_path (str): Dir path to the saved model.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(model_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)


def save_training_history(history, dir_path):
    """
    Save tensorflow model training history to a JSON file
    """
    hist_df = pd.DataFrame(history.history)
    hist_json_file = os.path.join(dir_path, HISTORY_FNAME)
    with open(hist_json_file, mode="w") as file_:
        hist_df.to_json(file_)
