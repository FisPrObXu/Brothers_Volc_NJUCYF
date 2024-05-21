import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from statsmodels.nonparametric.smoothers_lowess import lowess

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
        # 交换图像的X轴和Y轴
        img = cv2.transpose(img)
        h, w = img.shape
        central_region = img[h//3:2*h//3, w//3:2*w//3]
        avg_gray = np.mean(central_region)
        depths.append(depth)
        avg_grays.append(avg_gray)
    return np.array(depths, dtype=np.float32).reshape(-1, 1), np.array(avg_grays)

def build_and_train_lowess_model(depths, avg_grays, frac=0.12):
    lowess_results = lowess(avg_grays, depths.flatten(), frac=frac)
    return lowess_results

def build_and_train_svm_model(depths, avg_grays):
    model = make_pipeline(StandardScaler(), SVR(C=100.0, epsilon=0.001))
    model.fit(depths, avg_grays.ravel())
    return model

def visualize_results(depths, avg_grays, lowess_results, svm_model):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()  # Get current axes
    ax.invert_yaxis()  # Invert y-axis to make depths increase as you go down
    ax.xaxis.tick_top()  # Move x-axis to top
    ax.xaxis.set_label_position('top')  # Label x-axis at top
    plt.scatter(depths, avg_grays, color='grey', label='Data Points')
    plt.plot(lowess_results[:, 0], lowess_results[:, 1], color='red', label='LOWESS Fit')
    
    svm_predictions = svm_model.predict(depths)
    plt.plot(depths, svm_predictions, color='blue', label='SVM Prediction')
    
    plt.xlabel('Depth')
    plt.ylabel('Average Gray Value')
    plt.title('Comparison of LOWESS and SVM Polynomial Regression Fit')
    plt.legend()
    plt.show()

def export_data_to_csv(depths, avg_grays, lowess_results, svm_predictions, filename="data.csv"):
    data = np.hstack((depths, avg_grays.reshape(-1, 1), lowess_results[:, 1].reshape(-1, 1), svm_predictions.reshape(-1, 1)))
    df = pd.DataFrame(data, columns=['Depth', 'Average_Gray', 'LOWESS_Prediction', 'SVM_Prediction'])
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

def print_svm_model_details(svm_model):
    svr = svm_model.named_steps['svr']
    support_vectors = svr.support_
    dual_coefs = svr.dual_coef_
    intercept = svr.intercept_

    print("Support Vectors Indices:")
    print(support_vectors)
    print("\nDual Coefficients:")
    print(dual_coefs)
    print("\nIntercept:")
    print(intercept)

def main():
    root_dir = 'C:/Users/XuuX/Desktop/Ocean Drilling Data/1530/ex-corephototrimnormalize'
    depths, avg_grays = load_images_and_prepare_data(root_dir)
    lowess_results = build_and_train_lowess_model(depths, avg_grays, frac=0.2)
    svm_model = build_and_train_svm_model(depths, avg_grays)
    visualize_results(depths, avg_grays, lowess_results, svm_model)
    svm_predictions = svm_model.predict(depths)
    export_data_to_csv(depths, avg_grays, lowess_results, svm_predictions)
    print_svm_model_details(svm_model)

if __name__ == '__main__':
    main()
