import pandas as pd
import numpy as np
import os
import typing
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


usa_west_time_series_file = 'data/time_series.csv'
usa_west_meta_data_file = 'data/metadata.csv'


raw_pm_series = pd.read_csv(usa_west_time_series_file,index_col='time')
raw_pm_series.index = pd.to_datetime(raw_pm_series.index)

print(raw_pm_series)

metadata = pd.read_csv(usa_west_meta_data_file,index_col='Unnamed: 0')
# raw_pm_series.index = pd.to_datetime(raw_pm_series.index)
print(metadata)

print(list(metadata.index))

valid_pm_series = raw_pm_series.dropna()
print(valid_pm_series.shape)

daily_pm_series = valid_pm_series.resample('D').mean()
print(daily_pm_series.shape)

print(daily_pm_series)

weather_files = ['Seattle','Portland','San Francisco']
weather_file = weather_files[1]+'.csv'

def eliminate_invalid_values(raw_data):
    raw_data.Temperature = raw_data.Temperature.mask(raw_data.Temperature < 0, np.nan).astype(float)
    raw_data['Dew Point'] = raw_data['Dew Point'].mask(raw_data['Dew Point'] < 0, np.nan).astype(float)
    raw_data.Pressure = raw_data.Pressure.mask(raw_data.Pressure == 0, np.nan).astype(float)
    raw_data.Humidity = raw_data.Humidity.mask(raw_data.Humidity == 0, np.nan).astype(float)
    return raw_data

def Prepare_data(raw_weather_series):
    weather_series_attributes = ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Pressure'] # 'Precip.' is ingnored because of almost all being zero
    weather_series_model = raw_weather_series[weather_series_attributes].resample('H').mean()
    weather_series_model['Dew Point'] = weather_series_model['Dew Point'].mask(weather_series_model['Dew Point'] < 0, np.nan).astype(float)
    return weather_series_model

def FillMissingDataFromDays(x, days=3):
    ss = [x.shift(shft, freq='D') for shft in np.delete(np.arange(-days, days + 1), days)]
    return x.fillna((pd.concat(ss, axis=1).mean(axis=1)))

def read_and_prepare_weather_file(weather_file):
  raw_weather_series = pd.read_csv(weather_file,index_col='Time')
  raw_weather_series.index = pd.to_datetime(raw_weather_series.index)
  raw_weather_series.index.name = 'time'
  return raw_weather_series

raw_weather_series = read_and_prepare_weather_file(weather_file)
raw_weather_series.info()

print(raw_weather_series)

weather_series_model = Prepare_data(raw_weather_series)
weather_series_model.describe()

meteodata_dict = {}

for zone in weather_files:
  weather_file = zone+'.csv'
  raw_weather_series = read_and_prepare_weather_file(weather_file)
  weather_series_model = Prepare_data(raw_weather_series)
  weather_series_model = weather_series_model.apply(FillMissingDataFromDays, args=[3])
  meteodata_dict[zone] = weather_series_model.resample('D').mean()

print(meteodata_dict)

print(weather_series_model)

# fig = px.line(weather_series_model)
# fig.show()

print(weather_series_model.info())

"""# Model Data Defining"""

weather_array = daily_pm_series

print(weather_array)

from geopy.distance import geodesic

arr = []

for ids,source in metadata.iterrows():
  a = []
  for idd,destination in metadata.iterrows():
    dist = (geodesic((source.Latitude,source.Longitude), (destination.Latitude,destination.Longitude)).kilometers)
    a.append(dist)
  arr.append(a)

route_distances = pd.DataFrame(arr,index=metadata.index,columns=metadata.index)
# route_distances = route_distances.loc[state_zones,state_zones]  # Meteo
print(route_distances)

"""## Misc

# Model
"""

train_size, val_size = 0.6, 0.2


def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
    

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    train_array = data_array[:num_train]
    mean, std = train_array.mean(axis=0), train_array.std(axis=0)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train : (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val) :] - mean) / std

    return train_array, val_array, test_array


train_array, val_array, test_array = preprocess(weather_array, train_size, val_size)

print(f"train set size: {train_array.shape}")
print(f"validation set size: {val_array.shape}")
print(f"test set size: {test_array.shape}")

from tensorflow.keras.preprocessing import timeseries_dataset_from_array

batch_size = 64
input_sequence_length = 14
forecast_horizon = 7
multi_horizon = True


def create_tf_dataset(
    data_array: np.ndarray,
    input_sequence_length: int,
    forecast_horizon: int,
    batch_size: int = 128,
    shuffle=True,
    multi_horizon=True,
):
    
    inputs = timeseries_dataset_from_array(
        np.expand_dims(data_array[:-forecast_horizon], axis=-1),
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        data_array[target_offset:],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    if shuffle:
        dataset = dataset.shuffle(100)

    return dataset.prefetch(16).cache()


train_dataset, val_dataset = (
    create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
    for data_array in [train_array, val_array]
)

test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0],
    shuffle=False,
    multi_horizon=multi_horizon,
)

print(train_dataset)

# computing the adjaccency matrix using geo desic distance
def compute_adjacency_matrix(
    route_distances: np.ndarray, sigma2: float, epsilon: float
):
    
    num_routes = route_distances.shape[0]
    route_distances = route_distances / 10000.0
    w2, w_mask = (
        route_distances * route_distances,
        np.ones([num_routes, num_routes]) - np.identity(num_routes),
    )
    return (np.exp(-w2 / sigma2) >= epsilon) * w_mask

class GraphInfo:
    def __init__(self, edges: typing.Tuple[list, list], num_nodes: int):
        self.edges = edges
        self.num_nodes = num_nodes


sigma2 = 0.1
epsilon = 0.5
adjacency_matrix = compute_adjacency_matrix(route_distances, sigma2, epsilon)
node_indices, neighbor_indices = np.where(adjacency_matrix == 1)
graph = GraphInfo(
    edges=(node_indices.tolist(), neighbor_indices.tolist()),
    num_nodes=adjacency_matrix.shape[0],
)
print(f"number of nodes: {graph.num_nodes}, number of edges: {len(graph.edges[0])}")

# defining the architecture
class GraphConv(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        graph_info: GraphInfo,
        aggregation_type="mean",
        combination_type="concat",
        activation: typing.Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.graph_info = graph_info
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.weight = tf.Variable(
            initial_value=keras.initializers.glorot_uniform()(
                shape=(in_feat, out_feat), dtype="float32"
            ),
            trainable=True,
        )
        self.activation = layers.Activation(activation)

    def aggregate(self, neighbour_representations: tf.Tensor):
        aggregation_func = {
            "sum": tf.math.unsorted_segment_sum,
            "mean": tf.math.unsorted_segment_mean,
            "max": tf.math.unsorted_segment_max,
        }.get(self.aggregation_type)

        if aggregation_func:
            return aggregation_func(
                neighbour_representations,
                self.graph_info.edges[0],
                num_segments=self.graph_info.num_nodes,
            )

        raise ValueError(f"Invalid aggregation type: {self.aggregation_type}")

    def compute_nodes_representation(self, features: tf.Tensor):        
        return tf.matmul(features, self.weight)

    def compute_aggregated_messages(self, features: tf.Tensor):
        neighbour_representations = tf.gather(features, self.graph_info.edges[1])
        aggregated_messages = self.aggregate(neighbour_representations)
        return tf.matmul(aggregated_messages, self.weight)

    def update(self, nodes_representation: tf.Tensor, aggregated_messages: tf.Tensor):
        if self.combination_type == "concat":
            h = tf.concat([nodes_representation, aggregated_messages], axis=-1)
        elif self.combination_type == "add":
            h = nodes_representation + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        return self.activation(h)

    def call(self, features: tf.Tensor):
        nodes_representation = self.compute_nodes_representation(features)
        aggregated_messages = self.compute_aggregated_messages(features)
        return self.update(nodes_representation, aggregated_messages)

class LSTMGC(layers.Layer):
    def __init__(
        self,
        in_feat,
        out_feat,
        lstm_units: int,
        input_seq_len: int,
        output_seq_len: int,
        graph_info: GraphInfo,
        graph_conv_params: typing.Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # graph conv layer
        if graph_conv_params is None:
            graph_conv_params = {
                "aggregation_type": "mean",
                "combination_type": "concat",
                "activation": None,
            }
        self.graph_conv = GraphConv(in_feat, out_feat, graph_info, **graph_conv_params)

        self.lstm = layers.LSTM(lstm_units, activation="relu")
        self.dense = layers.Dense(output_seq_len)

        self.input_seq_len, self.output_seq_len = input_seq_len, output_seq_len

    def call(self, inputs):

        # convert shape to  (num_nodes, batch_size, input_seq_len, in_feat)
        inputs = tf.transpose(inputs, [2, 0, 1, 3])

        gcn_out = self.graph_conv(
            inputs
        )  # gcn_out has shape: (num_nodes, batch_size, input_seq_len, out_feat)
        shape = tf.shape(gcn_out)
        num_nodes, batch_size, input_seq_len, out_feat = (
            shape[0],
            shape[1],
            shape[2],
            shape[3],
        )

        # LSTM takes only 3D tensors as input
        gcn_out = tf.reshape(gcn_out, (batch_size * num_nodes, input_seq_len, out_feat))
        lstm_out = self.lstm(
            gcn_out
        )  # lstm_out has shape: (batch_size * num_nodes, lstm_units)

        dense_output = self.dense(
            lstm_out
        )  # dense_output has shape: (batch_size * num_nodes, output_seq_len)
        output = tf.reshape(dense_output, (num_nodes, batch_size, self.output_seq_len))
        return tf.transpose(
            output, [1, 2, 0]
        )  # returns Tensor of shape (batch_size, output_seq_len, num_nodes)

in_feat = 1
batch_size = 64
epochs = 20
input_sequence_length = 14
forecast_horizon = 7
multi_horizon = False
out_feat = 10
lstm_units = 64
graph_conv_params = {
    "aggregation_type": "mean",
    "combination_type": "concat",
    "activation": None,
}

st_gcn = LSTMGC(
    in_feat,
    out_feat,
    lstm_units,
    input_sequence_length,
    forecast_horizon,
    graph,
    graph_conv_params,
)
inputs = layers.Input((input_sequence_length, graph.num_nodes, in_feat))
outputs = st_gcn(inputs)

model = keras.models.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=0.0002),
    loss=keras.losses.MeanSquaredError(),
)
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[keras.callbacks.EarlyStopping(patience=10)],
)

print(inputs)

print(model)

print(test_array)

print(y[:, 0, -1].shape)

x_test, y = next(test_dataset.as_numpy_iterator())
y_pred = model.predict(x_test)
plt.figure(figsize=(18, 6))
plt.plot(y[:, 0, -1])
plt.plot(y_pred[:, 0, -1])
plt.legend(["actual", "forecast"])

naive_mse, model_mse = (
    np.square(x_test[:, -1, :, 0] - y[:, 0, :]).mean(),
    np.square(y_pred[:, 0, :] - y[:, 0, :]).mean(),
)
print(f"naive MAE: {naive_mse}, model MAE: {model_mse}")

iterable = test_dataset.as_numpy_iterator()
print(iterable)
# np.fromiter(iterable, float)

print(x_test.shape, y.shape)

print(y.shape,y_pred.shape)

for x_test, y in (test_dataset.as_numpy_iterator()):
  print(x_test.shape, y.shape)

from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(pd.Series(y[:, 0, 0]).fillna(0), pd.Series(y_pred[:, 0, 0]).fillna(0), squared=False)
print(rmse)

print(x_test.shape)

# y_flat,y_pred_flat = y.reshape(-1,7),y_pred.reshape(-1,7)
y_flat,y_pred_flat = y[:,:,-1],y_pred[:,:,-1]
# y_flat,y_pred_flat = pd.DataFrame(y_flat),pd.DataFrame(y_pred_flat)

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_scaled_error(y, y_hat, y_train):
    
    ## Naive in-sample Forecast
    # naive_y_hat = y_train[:-1]
    # naive_y = y_train[1:]

    naive_y_hat = y_train.values[:-1]
    naive_y = y_train.values[1:]
    # print(y_train.shape,naive_y_hat,naive_y)

    ## Calculate MAE (in sample)
    mae_in_sample = np.mean(np.abs(naive_y - naive_y_hat))
    mae = np.mean(np.abs(y - y_hat))
    
    return mae/mae_in_sample

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,median_absolute_error
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def get_performance_metrices(y_test,predictions,model_name):
  eval_matrices = ['model','mae','mse','rmse','mape','mdae','mase'][:-1]

  metrices = pd.DataFrame(index=range(7),columns=eval_matrices)
  # print(metrices)

  for i in range(7):
    real = y_flat[i]
    # real.index = real.index.shift(i)
    prediction = y_pred_flat[i]
    mae = np.round(mean_absolute_error(real, prediction), 3)
    mse = np.round(mean_squared_error(real, prediction), 3)
    rmse = np.round(mean_squared_error(real, prediction,squared=False), 3)
    mape = np.round(mean_absolute_percentage_error(real, prediction), 3)
    mdae = np.round(median_absolute_error(real, prediction), 3)
    # r2 = np.round(r2_score(real, prediction), 3)
    # mase = np.round(mean_absolute_scaled_error(real, prediction,y_train), 3)
    metrices.loc[i] = [model_name,mae,mse,rmse,mape,mdae]
  return metrices

def get_models_metrices_concatanated_indexed(models_metrices_list):
  models_metrices_concatanated = pd.concat(models_metrices_list)
  models_metrices_concatanated.index.name = 'time_offset'
  models_metrices_concatanated_indexed = models_metrices_concatanated.set_index([models_metrices_concatanated.index,'model'])
  return models_metrices_concatanated_indexed

perf = get_performance_metrices(y_flat,y_pred_flat,"GNN")
perf.to_csv("GNN.csv")
