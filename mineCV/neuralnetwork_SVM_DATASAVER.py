import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def scheduler(epoch, lr):
    min_lr = 0.0001
    if epoch < 100:
        return max(lr, min_lr)
    else:
        return max(lr * np.exp(-0.01), min_lr)

class PolynomialLayer(tf.keras.layers.Layer):
    def __init__(self, degree, **kwargs):
        super(PolynomialLayer, self).__init__(**kwargs)
        self.degree = degree

    def build(self, input_shape):
        self.coefficients = self.add_weight(
            name='coefficients',
            shape=(self.degree + 1,),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        x = inputs
        outputs = self.coefficients[0]  # constant term
        for i in range(1, self.degree + 1):
            outputs += self.coefficients[i] * tf.pow(x, i)
        return outputs
    
def load_images_and_prepare_data(root_dir):
    img_names = os.listdir(root_dir)
    img_names.sort(key=lambda x: int(x.split('A')[1].split('_')[0]))

    depths = []
    avg_grays = []

    for name in img_names:
        depth = int(name.split("A")[1].split("_")[0])
        img_path = os.path.join(root_dir, name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        h, w = img.shape
        central_region = img[h//3:2*h//3, w//3:2*w//3]
        avg_gray = np.mean(central_region)

        depths.append(depth)
        avg_grays.append(avg_gray)

    return np.array(depths, dtype=np.float32).reshape(-1, 1), np.array(avg_grays, dtype=np.float32)

def prepare_polynomial_features(depths, degree=3):
    n = len(depths)
    features = np.ones((n, degree))
    for i in range(1, degree):
        features[:, i] = depths[:, 0] ** (i + 1)
    return features

def build_and_train_polynomial_regression_model(features, targets, epochs=2000, degree=3):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='relu', input_shape=(degree,)),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    history = model.fit(features, targets, epochs=epochs, verbose=0, validation_split=0.4)
    return model, history

def build_and_train_svm_model(features, targets, degree=3):
    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', degree=degree, C=100.0, epsilon=10))
    model.fit(features, targets.ravel())
    return model

def visualize_results(depths, avg_grays, nn_model, svm_model, features):
    plt.figure(figsize=(10, 6))
    plt.scatter(depths, avg_grays, color='gray', label='Data Points')

    x_values = np.linspace(min(depths), max(depths), 500).reshape(-1, 1)
    x_features = prepare_polynomial_features(x_values, features.shape[1])
    nn_y_values = nn_model.predict(x_features)
    svm_y_values = svm_model.predict(x_features)

    plt.plot(x_values, nn_y_values, color='red', label='NN Polynomial Regression Prediction')
    plt.plot(x_values, svm_y_values, color='blue', label='SVM Polynomial Regression Prediction')

    plt.xlabel('Depth')
    plt.ylabel('Average Gray Value')
    plt.title('Comparison of Polynomial Regression Fits')
    plt.legend()
    plt.show()

def visualize_training_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def export_data_to_csv(depths, avg_grays, filename="data.csv"):
    # 确保两个数组的形状相同
    depths = depths.reshape(-1, 1)  # 确保depths是(n, 1)形状
    avg_grays = avg_grays.reshape(-1, 1)  # 确保avg_grays也是(n, 1)形状

    # 将depths和avg_grays堆叠成一个二维数组
    data = np.hstack((depths, avg_grays))
    df = pd.DataFrame(data, columns=['Depth', 'Average_Gray'])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

if __name__ == '__main__':
    root_dir = './ex-corephototrimnormalize'
    depths, avg_grays = load_images_and_prepare_data(root_dir)
    degree = 3  # 多项式的度数
    features = prepare_polynomial_features(depths, degree)

    # 训练神经网络
    nn_model, history = build_and_train_polynomial_regression_model(features, avg_grays, degree=degree)
    visualize_training_history(history)

    # 训练SVM
    svm_model = build_and_train_svm_model(features, avg_grays, degree=degree)

    # 可视化比较结果
    visualize_results(depths, avg_grays, nn_model, svm_model, features)

    # 导出数据
    export_data_to_csv(depths, avg_grays)
