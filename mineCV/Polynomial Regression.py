import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split

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
    return np.array(depths, dtype=np.float32).reshape(-1, 1), np.array(avg_grays)

def build_and_train_nn_model(depths, avg_grays, degree=4, epochs=1000, batch_size=16, learning_rate=0.0001):
    poly = PolynomialFeatures(degree)
    depths_poly = poly.fit_transform(depths)
    
    # 数据归一化
    scaler = StandardScaler()
    depths_poly = scaler.fit_transform(depths_poly)
    
    # 训练集和验证集划分
    depths_poly_train, depths_poly_val, avg_grays_train, avg_grays_val = train_test_split(depths_poly, avg_grays, test_size=0.2, random_state=42)
    
    model = Sequential([
        Dense(256, input_dim=depths_poly.shape[1], activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    model.fit(depths_poly_train, avg_grays_train, validation_data=(depths_poly_val, avg_grays_val), epochs=epochs, batch_size=batch_size, verbose=1)
    return model, poly, scaler

def visualize_results(depths, avg_grays, nn_model, poly, scaler):
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')

    plt.scatter(depths, avg_grays, color='grey', label='Data Points')
    depths_poly = poly.transform(depths)
    depths_poly = scaler.transform(depths_poly)
    nn_predictions = nn_model.predict(depths_poly)
    plt.plot(depths, nn_predictions, color='blue', label='Neural Network Prediction')

    plt.ylabel('Average Gray Value')
    plt.xlabel('Depth')
    plt.title('Neural Network Polynomial Regression Fit')
    plt.legend()
    plt.show()

def export_data_to_csv(depths, avg_grays, nn_predictions, filename="data.csv"):
    data = np.hstack((depths, avg_grays.reshape(-1, 1), nn_predictions))
    df = pd.DataFrame(data, columns=['Depth', 'Average_Gray', 'NN_Prediction'])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def print_nn_model_equation(model, poly, scaler):
    feature_names = poly.get_feature_names_out()
    
    print("Neural Network Model Equation:")
    for i, layer in enumerate(model.layers):
        weights = layer.get_weights()
        if weights:  # 检查该层是否有权重
            weight, bias = weights
            print(f"Layer {i+1}:")
            print(f"Weights: {weight}")
            print(f"Biases: {bias}")
            print()

def main():
    root_dir = 'C:/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephototrimnormalize'
    depths, avg_grays = load_images_and_prepare_data(root_dir)
    nn_model, poly, scaler = build_and_train_nn_model(depths, avg_grays, degree=4, epochs=1000, batch_size=16, learning_rate=0.0001)
    visualize_results(depths, avg_grays, nn_model, poly, scaler)
    depths_poly = poly.transform(depths)
    depths_poly = scaler.transform(depths_poly)
    nn_predictions = nn_model.predict(depths_poly)
    export_data_to_csv(depths, avg_grays, nn_predictions)
    print_nn_model_equation(nn_model, poly, scaler)

if __name__ == '__main__':
    main()
