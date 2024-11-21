import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier

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
    for data_load in data_list:
        with open(os.path.join('logs', data_load, 'data.pkl'), 'rb') as f:
            data += pickle.load(f)

    X = []
    y = []
    for observation, action in data:
        features = extract_features(observation)
        X.append(features)
        y.append(action)

    X = np.array(X)
    y = np.array(y)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X, y)

    env = AliensEnvPygame(level=0, render=False)

    with open(f'{env.log_folder}/gameplay_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print("模型训练完成")

if __name__ == '__main__':
    main()
