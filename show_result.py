# @Time : 2023/12/13
# @Author : WangXuSheng
import h5py
import numpy as np
import spectral
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, recall_score, cohen_kappa_score, classification_report
from utils_HSI import open_file


def show_XA02(path):
    result = loadmat(path)
    results = result['results']
    prediction = (results["prediction"][0][0] + 1).flatten()
    target_label = loadmat("./datasets/XA02/XA02_gt_0.mat")['XA_gt']
    _target_label = target_label
    target_label = target_label.reshape(1, -1).flatten()
    _prediction = prediction
    for index in np.where(target_label == 0)[0]:
        _prediction = np.insert(_prediction, index, 0)
    _prediction = _prediction.reshape(_target_label.shape)

    spectral.save_rgb("./results_imgs/XA02_2_pred.jpg",
                      _prediction,
                      colors=spectral.spy_colors)
    class_name = ['Road', 'Building', 'Tree', 'Farmland', 'Bare Land', 'Orchard', 'Water']
    # classfication report
    test_pred = (results["prediction"][0][0] + 1).flatten()
    test_true = (results["label"][0][0] + 1).flatten()

    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average='macro')
    kappa = cohen_kappa_score(test_true, test_pred)
    report_log = F"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n"
    report_log += classification_report(test_true, test_pred, target_names=class_name, digits=4)
    print(report_log)


def show_Pavia(path):
    result = loadmat(path)
    results = result['results']
    prediction = (results["prediction"][0][0] + 1).flatten()
    target_label = loadmat("./datasets/Pavia/paviaC_7gt.mat")['map']
    _prediction = np.zeros_like(target_label)
    x, y = np.where(target_label > 0)
    for index in range(np.sum(target_label > 0)):
        _prediction[x[index]][y[index]] = prediction[index]
    spectral.save_rgb("./results_imgs/PaviaC_pred.jpg",
                      _prediction,
                      colors=spectral.spy_colors)

    class_name = ["tree", "asphalt", "brick",
                  "bitumen", "shadow", 'meadow', 'bare soil']
    # classfication report
    test_pred = (results["prediction"][0][0] + 1).flatten()
    test_true = (results["label"][0][0] + 1).flatten()

    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average='macro')
    kappa = cohen_kappa_score(test_true, test_pred)
    report_log = F"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n"
    report_log += classification_report(test_true, test_pred, target_names=class_name, digits=4)
    print(report_log)


def show_Houston(path):
    result = loadmat(path)
    results = result['results']
    # prediction img
    prediction = (results["prediction"][0][0] + 1).flatten()
    target_label = h5py.File("./datasets/Houston/Houston18_7gt.mat", 'r')['map'][:]
    target_label = np.transpose(target_label, (1, 0))
    _target_label = target_label
    target_label = target_label.reshape(1, -1).flatten()
    _prediction = prediction
    for index in np.where(target_label == 0)[0]:
        _prediction = np.insert(_prediction, index, 0)

    _prediction = _prediction.reshape(_target_label.shape)
    spectral.save_rgb("./results_imgs/Houston18_pred.jpg",
                      _prediction,
                      colors=spectral.spy_colors)
    class_name = ["grass healthy", "grass stressed", "trees",
                  "water", "residential buildings",
                  "non-residential buildings", "road"]
    # classfication report
    test_pred = (results["prediction"][0][0] + 1).flatten()
    test_true = (results["label"][0][0] + 1).flatten()

    OA = accuracy_score(test_true, test_pred)
    AA = recall_score(test_true, test_pred, average='macro')
    kappa = cohen_kappa_score(test_true, test_pred)
    report_log = F"OA: {OA}\nAA: {AA}\nKappa: {kappa}\n"
    report_log += classification_report(test_true, test_pred, target_names=class_name, digits=4)
    print(report_log)


path = ""

result = loadmat(path)
results = result['results']
OA = results['Accuracy'][0][0][0][0]
AA = results['AA'][0][0][0][0]
Kappa = results['Kappa'][0][0][0][0]
print(f'OA: {OA:.2f} AA: {AA * 100:.2f} Kappa: {Kappa * 100:.2f}')
# show_Houston(path)
# show_Pavia(path)
# show_XA02(path)
