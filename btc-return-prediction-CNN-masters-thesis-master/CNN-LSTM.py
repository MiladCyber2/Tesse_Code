import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from functions import *
from import_data import data_scaled, data_scalers, data_raw


WINDOW = 50
HORIZON = 1
SPLIT = 0.8
EPOCHS = 100
BATCH_SIZE = 32


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def save_metrics_to_file(y_true, y_pred, file_path):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics = {
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1]
    }
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")

def plot_confusion_matrix(cm, title='Confusion Matrix'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()

def main():
    ################### Data pre processing ######################
    data = data_scaled
    x_train, y_train, x_test, y_test = create_train_test(
        data,
        {"btc_logreturn": data["btc_logreturn"]},
        window=WINDOW,
        horizon=HORIZON,
        split=SPLIT,
    )

    n_features = len(data.keys())

    # Define the input layer
    inputs = tf.keras.Input(shape=[WINDOW, n_features])

    # Define the convolutional layers
    conv_layer = tf.keras.layers.Conv1D(
        filters=32, kernel_size=3, padding="same", activation=tf.nn.relu
    )(inputs)
    maxpool_layer = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv_layer)
    conv_layer = tf.keras.layers.Conv1D(
        filters=64, kernel_size=3, padding="same", activation=tf.nn.relu
    )(maxpool_layer)
    maxpool_layer = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv_layer)
    conv_layer = tf.keras.layers.Conv1D(
        filters=128, kernel_size=3, padding="same", activation=tf.nn.relu
    )(maxpool_layer)
    maxpool_layer = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2)(conv_layer)

    # Flatten the output of the convolutional layers
    # flatten_layer = tf.keras.layers.Flatten()(maxpool_layer)

    # Define the LSTM layers
    lstm_layer = tf.keras.layers.LSTM(
        units=1024, return_sequences=True, activation=tf.nn.relu
    )(maxpool_layer)
    lstm_layer = tf.keras.layers.LSTM(units=512, activation=tf.nn.relu)(lstm_layer)

    # Define the output layer
    outputs = tf.keras.layers.Dense(units=1)(lstm_layer)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.summary()
    tf.keras.utils.plot_model(
        model, to_file="C:/Users/Syber/Desktop/VS_code/code/btc-return-prediction-CNN-masters-thesis-master/Plots/model_shape_CNN-LSTM.png", show_shapes=True
    )
    #
    # print(model.summary())
    # stop early if model is not improving for patience=n epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=15, restore_best_weights=True
    )
    model.compile(loss="mae", optimizer="adam", metrics=["mse", "mae"])
    model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        callbacks=[es_callback],
        shuffle=False,
    )

    epoch_count = len(model.history.epoch)
    model_history = pd.DataFrame(model.history.history)
    model_history["epoch"] = model.history.epoch
    num_epochs = model_history.shape[0]
    # plot_model_loss_training(model)

    # get the models predicted values
    # pred_train = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_train))
    pred_test = data_scalers["btc_logreturn"].inverse_transform(model.predict(x_test))
    y_test_unscaled = data_scalers["btc_logreturn"].inverse_transform(y_test[:, :, 0])
    
    # Binarize the predictions and true values for classification metrics
    y_test_binary = (y_test_unscaled > 0).astype(int)
    y_pred_binary = (pred_test > 0).astype(int)

    # Save metrics to file
    save_metrics_to_file(y_test_binary, y_pred_binary, 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_metrics.csv')

    # Save confusion matrix data
    cm = confusion_matrix(y_test_binary, y_pred_binary)
    cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
    cm_df.to_csv('C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_confusion_matrix_data.csv')
    print("Confusion matrix saved to C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_confusion_matrix_data.csv")

    # Plot confusion matrix
    plot_confusion_matrix(cm)

    model_stats(model.name, pred_test, x_test, y_test_unscaled, model, epoch_count)
    print_model_statistics(
        model_statistics(pred_test, x_test, y_test, model), x_test, y_test, model
    )

    save_loss_history(
        num_epochs, model_history["loss"], model_history["val_loss"], "last_model"
    )
    main_plot(
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
    )
    pred_plot(
        "CNN_LSTM",
        WINDOW,
        HORIZON,
        EPOCHS,
        model,
        x_train,
        num_epochs,
        model_history,
        data,
        data_scalers,
        pred_test,
    )
    return


if __name__ == "__main__":
    main()
