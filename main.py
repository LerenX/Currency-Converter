import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import sys


class Converter(object):

    def __init__(self):
        self.data = pd.read_csv("eurofxref-hist\\eurofxref-hist.csv", sep=",")
        self.data = self.data[
            ["Date", "USD", "JPY", "BGN", "CYP", "CZK", "DKK", "EEK", "GBP", "HUF", "LTL", "LVL", "MTL", "PLN", "ROL",
             "RON", "SEK", "SIT", "SKK", "CHF", "ISK", "NOK", "HRK", "RUB", "TRL", "TRY", "AUD", "BRL", "CAD", "CNY",
             "HKD", "IDR", "ILS", "INR", "KRW", "MXN", "MYR", "NZD", "PHP", "SGD", "THB", "ZAR", ]]

    def convert_currency(self):
        try:
            userInput = input("CURRENCY: (AMOUNT_Y-M-D_CURRENCY1_CURRENCY2) ex: (1000_2020-02-14_SEK_USD)\n").split("_")
            AMOUNT, DATE, CURRENCY1, CURRENCY2 = int(userInput[0]), userInput[1], userInput[2], userInput[3]
            for i, col in enumerate(self.data["Date"]):
                if DATE in self.data["Date"][i]:
                    break
            c1, c2 = self.data[CURRENCY1][i], self.data[CURRENCY2][i]
            finalC2 = c2 * (AMOUNT / c1)
            print("{}: {} {} = {:.2f} {}".format(DATE, AMOUNT, CURRENCY1, finalC2, CURRENCY2))
        except Exception as e:
            print(e)
            converter.convert_currency()

    def visualize_currency(self):
        data_list = []
        try:
            CURRENCY = input("CURRENCY:\n")
        except Exception as e:
            print(e)
            converter.visualize_currency()
        for i, col in enumerate(self.data[CURRENCY]):
            data_list.append(self.data[CURRENCY][i])
        data_list.reverse()
        style.use("ggplot")
        plt.plot(data_list)
        plt.ylabel(f"{CURRENCY} per 1 EUR")
        plt.show()

    def predict_future_currency(self):
        try:
            CURRENCY = input("CURRENCY:\n")
        except Exception as e:
            print(e)
            converter.predict_future_currency()
        X = np.array(self.data.drop([CURRENCY], 1))
        y = np.array(self.data[CURRENCY])
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
        best = 0
        for _ in range(10000):
            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)
            linear = linear_model.LinearRegression()
            linear.fit(x_train, y_train)
            acc = linear.score(x_test, y_test)
            print(acc)
            if acc > best:
                best = acc
                with open("studentmodel.pickle", "wb") as f:
                    pickle.dump(linear, f)
        print("Best: ", best)
        """
        pickle_in = open("studentmodel.pickle", "rb")
        linear = pickle.load(pickle_in)
        """
        print("Co: ", linear.coef_)
        print("Intercept: ", linear.intercept_)
        predictions = linear.predict(x_test)
        for i in range(len(predictions)):
            print(predictions[i], x_test[i], y_test[i])


if __name__ == "__main__":
    converter = Converter()
    while True:
        user_input = input("Graph, Convert or Predict? (G/C/P) or Q for QUIT\n").lower()
        if user_input == "q":
            sys.exit()
        elif user_input == "g":
            converter.visualize_currency()
            while input("Continue? (Y/N)\n").lower() == "y":
                converter.visualize_currency()
        elif user_input == "p":
            converter.predict_future_currency()
            while input("Continue? (Y/N)\n").lower() == "y":
                converter.predict_future_currency()
        else:
            converter.convert_currency()
            while input("Continue? (Y/N)\n").lower() == "y":
                converter.convert_currency()
