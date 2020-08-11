import argparse
import time
import os
import sys
from pandas import read_csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers
import tensorflow as tf
from shutil import copyfile


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128, help='Batch Size during training [default: 128]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--step_per_epoch', type=int, default=50, help='Step per epoch [default: 50]')
parser.add_argument('--num_parameter', type=int, default=5, help='# of parameter [default: 5]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--cycle', type=int, default=300, help='# of sliding window [default: 300]')
parser.add_argument('--num_output', type=int, default=2, help='# of output [default: 2]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--model_path', default='', help='model file path. ex)./my_model.h5 [default:]')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
STEP_PER_EPOCH = FLAGS.step_per_epoch
OPTIMIZER = FLAGS.optimizer
NUM_PARAMETER = FLAGS.num_parameter
CYCLE = FLAGS.cycle
NUM_OUTPUT = FLAGS.num_output
LOG_DIR = FLAGS.log_dir
MODEL_PATH = FLAGS.model_path

TRAIN_FILES = './datasets/Training.csv'
EVALUATE_FILES = './datasets/Evaluating.csv'
PREDICT_FILES = './datasets/Subject_1.csv'

def make_folder(time, root=""):
    path = root + "%04d-%02d-%02d_%02d_%02d_%02d" % (
        time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec)
    if not os.path.isdir(path):
        os.mkdir(path)
        return path


def sample_data(dataset, indexset, size):
    sample_data_list = []
    index_list = []

    for i in range(CYCLE, size):
        data = dataset[i - CYCLE: i, :]
        indices = indexset[i - CYCLE: i, :]
        sample_data_list.append(np.reshape(data, (CYCLE, NUM_PARAMETER)))
        index_list.append(np.reshape(indices, (CYCLE, NUM_OUTPUT)))
    sample_data = np.array(sample_data_list)
    index_data = np.array(index_list)

    return sample_data, index_data


def parameters(path, names):
    dataframe = read_csv(path, header=0)
    for i in range(len(names)):
        dataframe = dataframe.drop(names[i], axis=1)
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    return dataset[:, 0:NUM_OUTPUT], dataset[:, NUM_OUTPUT:]



def create_model():
    xInput = layers.Input(batch_shape=(None, CYCLE, NUM_PARAMETER))
    xLstm_1 = layers.LSTM(units=256, return_sequences=True)(xInput)
    xLstm_2 = layers.Bidirectional(layers.LSTM(units=128, return_sequences=True))(xLstm_1)
    xLstm_3 = layers.LSTM(units=128, return_sequences=True)(xLstm_2)
    xLstm_4 = layers.Bidirectional(layers.LSTM(units=64, dropout=0.2, return_sequences=True))(xLstm_3)
    xOutput = layers.Dense(NUM_OUTPUT)(xLstm_4)
    model = tf.keras.Model(xInput, xOutput)
    model.compile(loss='mse', optimizer=OPTIMIZER)
    return model


eliminate_parameter = ['Force_heel']
NUM_PARAMETER -= len(eliminate_parameter)

index_training, dataset_training = parameters(TRAIN_FILES, eliminate_parameter)
scaler = MinMaxScaler(feature_range=(0, 1))
data_training = scaler.fit_transform(dataset_training)
train_size = int(len(data_training))


def train():
    index_evaluate, dataset_evaluate = parameters(EVALUATE_FILES, eliminate_parameter)
    data_evaluating = scaler.transform(dataset_evaluate)
    evaluate_size = int(len(dataset_evaluate))

    train_data, train_index = sample_data(data_training, index_training, train_size)
    dataset_train = tf.data.Dataset.from_tensor_slices((train_data, train_index))
    dataset_train = dataset_train.repeat()
    dataset_train = dataset_train.shuffle(buffer_size=(int(train_size * 0.4) + 3 * BATCH_SIZE))
    dataset_train = dataset_train.batch(BATCH_SIZE)

    evaluation_data, evaluation_index = sample_data(data_evaluating, index_evaluate, evaluate_size)
    dataset_eval = tf.data.Dataset.from_tensor_slices((evaluation_data, evaluation_index))
    dataset_eval = dataset_eval.batch(BATCH_SIZE)

    path = make_folder(time.localtime())

    f = open(path + "/log.txt", 'w')
    sys.stdout = f

    copyfile("gait_lstm.py", path + "/gait_lstm.py")

    model = create_model()
    print(model.summary())
    cbs = [tf.keras.callbacks.ModelCheckpoint(filepath=path + '/checkpoint.keras', monitor='val_loss',
                                              verbose=1, save_weights_only=True, save_best_only=True)]
    model.fit(dataset_train, validation_data=dataset_eval, epochs=MAX_EPOCH,
              steps_per_epoch=STEP_PER_EPOCH, verbose=2, callbacks=cbs)

    model.save(path, save_format='tf')
    model.save(path + "/my_model.h5")

def predict():
    index_predict, dataset_predict = parameters(PREDICT_FILES, eliminate_parameter)
    data_predicting = scaler.transform(dataset_predict)
    predict_size = int(len(data_predicting))
    predict_data, predict_index = sample_data(data_predicting, index_predict, predict_size)

    new_model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    start = time.time()
    y_pred = new_model.predict(predict_data)
    end = time.time()
    print('time:', (end - start))

    np.savetxt("predict.txt", y_pred[:, -1, :], fmt="%.8f")
    np.savetxt("origin.txt", predict_index[:, -1, :], fmt="%.8f")


if __name__ == "__main__":
    if MODEL_PATH == '':
        train()
    else:
        predict()
