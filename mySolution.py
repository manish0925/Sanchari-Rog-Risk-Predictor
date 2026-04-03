import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

if __name__ == "__main__":
    # Load dataset
    df = pd.read_excel('DataSet.xlsx')

    # Split into train/test
    train, test = data_split(df, 0.2)

    # Separate features (X) and target (y)
    X_train = train[['WaterLogging', 'CleanWater', 'HouseCleaning', 'Macchardani']]
    y_train = train['InfectionProb']

    X_test = test[['WaterLogging', 'CleanWater', 'HouseCleaning', 'Macchardani']]
    y_test = test['InfectionProb']

    # Train model
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    # Save model
    with open('model.pkl', 'wb') as file:
        pickle.dump(clf, file)

    print("✅ Model trained and saved successfully!")
