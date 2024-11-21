import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from play import AliensEnvPygame

def extract_features(observation):

    # TODO

    grid = observation
    features = []

    def cell_to_feature(cell):
        object_mapping = {
            'floor': 0,
            'wall': 1,
            'avatar': 2,
            'alien': 3,
            'bomb': 4,
            'portalSlow': 5,
            'portalFast': 6,
            'sam': 7,
            'base': 8
        }
        feature_vector = [0] * len(object_mapping)
        for obj in cell:
            index = object_mapping.get(obj, -1)
            if index >= 0:
                feature_vector[index] = 1
        return feature_vector

    for row in grid:
        for cell in row:
            cell_feature = cell_to_feature(cell)
            features.extend(cell_feature)

    return np.array(features)

def main():
    data_list = []

    with open('./logs/folders_list.txt', 'r') as f:
        for line in f: 
            line = line.strip()
            data_list.append(line)
    
    data = []

    print(f"开始训练模型，共有{len(data_list)}个数据集")

    for data_load in data_list:
        with open(os.path.join('logs', data_load, 'data.pkl'), 'rb') as f:
            data += pickle.load(f)

    X = []
    y = []

    print(f"共有{len(data)}条数据")

    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    # clf = RandomForestClassifier(n_estimators=100, random_state=0)
    # clf = SVC(kernel='linear', C=1.0, random_state=0)
    # clf = KNeighborsClassifier(n_neighbors=10)  # 选择 k=10
    clf = GaussianNB()
    clf.fit(X, y)

    env = AliensEnvPygame(level=0, render=False)

    model_dir = f'{env.model_folder}'  # 假设 env.model_folder 是目标目录
    os.makedirs(model_dir, exist_ok=True)  # 如果目录不存在，自动创建

    with open(f'{env.model_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")

if __name__ == '__main__':
    main()
