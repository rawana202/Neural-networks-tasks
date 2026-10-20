import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path):

    data = pd.read_csv(path)
    return data

def split_data(data,target, selected_classes):

    #to split the data evenly between the two classes
    features1 = data[target == selected_classes[0]]
    features2 = data[target == selected_classes[1]]
    target1 = target[target == selected_classes[0]]
    target2 = target[target == selected_classes[1]]

    X_train1, X_test1, y_train1, y_test1 = train_test_split(features1, target1, test_size=0.2, random_state=42)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(features2, target2, test_size=0.2, random_state=42)

    X_train = pd.concat([X_train1, X_train2], axis=0)
    X_test = pd.concat([X_test1, X_test2], axis=0)
    y_train = pd.concat([y_train1, y_train2], axis=0)
    y_test = pd.concat([y_test1, y_test2], axis=0)

    return X_train, X_test, y_train, y_test

def preprocess_data(data,selected_classes):

    data = data[data.iloc[:, 0].isin(selected_classes)].copy()

    target = data.iloc[:, 0]
    features = data.iloc[:, 1:]


    #split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(features, target, selected_classes)

    
    # Handle missing values (if any) fill with mean for numerical columns and mode for categorical columns
    for column in X_train.columns:
        if X_train[column].dtype == 'object':
            mode = X_train[column].mode()[0]
            X_train[column] = X_train[column].fillna(mode)
            X_test[column] = X_test[column].fillna(mode)
        else:
            mean = X_train[column].mean()
            X_train[column] = X_train[column].fillna(mean)
            X_test[column]  = X_test[column].fillna(mean)


    # Map categorical values to numerical values
    mapping = {
        'Torgersen': 1,
        'Biscoe': 2,
        'Dream': 3
    }

    X_train['OriginLocation'] = X_train['OriginLocation'].map(mapping)
    X_test['OriginLocation'] = X_test['OriginLocation'].map(mapping)

    # Normalize the features(z score normalization)
    mean = X_train.mean()
    std = X_train.std().replace(0, 1)

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std


    # Randomly shuffle the training data
    positions = pd.Series(range(len(X_train))).sample(
        frac=1, random_state=42
    ).to_numpy()

    X_train = X_train.iloc[positions]
    y_train = y_train.iloc[positions]

    y_train, y_test = preprocess_target(y_train, y_test, selected_classes)

    return X_train, X_test, y_train, y_test


def preprocess_target(y_train, y_test, selected_classes):

    mapping = {selected_classes[0]: 1,
                selected_classes[1]: -1
                }

    y_train = y_train.map(mapping)
    y_test = y_test.map(mapping)

    return y_train, y_test