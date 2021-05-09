import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import pickle


if __name__ == '__main__':
    df = pd.read_csv('models.csv')
    x = df.drop('class', axis=1)
    y = df['class']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x_train, y_train)
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, predict)

    with open("train.model", 'wb') as f:
        pickle.dump(model,f)

    exit()