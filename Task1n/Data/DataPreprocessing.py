import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def DataReading():
    data=pd.read_csv(r"H:\uni41\NN\NNTasks\Task1n\Data\penguins.csv")
    print(data.head())
    print(data.info())
    print(data.isnull().sum())
    print(data.duplicated().sum())
    return data

def DataVisualization(data):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, column in enumerate(data.columns[:6]):
        ax = axes[i]

        if pd.api.types.is_numeric_dtype(data[column]):
            # Histogram for numerical columns
            data[column].dropna().hist(
                bins=15, ax=ax, edgecolor="black"
            )
            ax.set_ylabel("Frequency")
        else:
            # Bar chart for categorical columns
            data[column].value_counts().plot(
                kind="bar", ax=ax
            )
            ax.set_ylabel("Frequency")
            ax.tick_params(axis="x", rotation=45)

        ax.set_title(column)

    plt.tight_layout()
    plt.show()

def DataCleaning(data,Features,Target):

    # 1- Encoding categorical data 
    encoder = LabelEncoder()
    data["OriginLocation"] = encoder.fit_transform(data["OriginLocation"])#the output of the encoder must be in one column cause the model will not accept more than two columns as input --> so the best sol is labelEncoder

    if Target=="Adelie & Chinstrap":  # cause the signum function output is -1 and 1 so we will map the classes to these values  
       
        data = data[data["Species"].isin(["Adelie", "Chinstrap"])]
        data["Species"] = data["Species"].map({
            "Adelie": 1,
            "Chinstrap": -1
        })

    elif Target=="Chinstrap & Gentoo":
        data = data[data["Species"].isin(["Chinstrap", "Gentoo"])]  
        data["Species"] = data["Species"].map({
            "Chinstrap": 1,
            "Gentoo": -1
        })

    elif Target=="Adelie & Gentoo":
        data = data[data["Species"].isin(["Adelie", "Gentoo"])]    
        data["Species"] = data["Species"].map({
            "Adelie": 1,
            "Gentoo": -1
        })

    data = data.reset_index(drop=True)   

    # Split each class: 30 train, 20 test

    class_1 = data[data["Species"] == 1]
    class_2 = data[data["Species"] == -1]

    train_1, test_1 = train_test_split(
        class_1, train_size=30, test_size=20, random_state=42
    )

    train_2, test_2 = train_test_split(
        class_2, train_size=30, test_size=20, random_state=42
    )

    # Combine the classes
    train_data = pd.concat([train_1, train_2]).sample(frac=1, random_state=42) #60
    test_data = pd.concat([test_1, test_2]).sample(frac=1, random_state=42)    #40

    x_train = train_data[Features]
    y_train = train_data["Species"].values

    x_test = test_data[Features]
    y_test = test_data["Species"].values

    # 2- Handle missing values by filling with the mean of each feature

    means = x_train.mean()
    x_train = x_train.fillna(means).values
    x_test = x_test.fillna(means).values

   # 3- Scaling the data

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test    

def DataPreprocessing(features, target_classes):
    data = DataReading()
    x_train, y_train, x_test, y_test = DataCleaning(data, features, target_classes)
    return x_train, y_train, x_test, y_test