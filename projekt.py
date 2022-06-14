import numpy as np
import pandas
from matplotlib import pyplot as plt
import seaborn
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pandas.read_excel("wypadki_dane.xlsx")
df.index = df["rok"]
df = df.drop(columns="rok")
plt.ticklabel_format(useOffset=False)

class CustomModelWrapper:
    def __init__(self, pred_fun, params):
        self.pred_fun = pred_fun
        self.params = params

    def predict(self, x):
        return self.pred_fun(x.ravel(), *self.params)

def func(x, a, b, c):
    return a * x * x + b * x + c

def new_plot(x, y, xlabel, title, filename):
    plt.clf()

    X = x.values
    Y = y.values

    X_axis = np.linspace(start=X.min(), stop=X.max(), num=np.floor(X.max()-X.min()).astype(np.uint32))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)

    model_lin = LinearRegression()
    model_lin.fit(X_train, Y_train)

    params, _ = curve_fit(func, xdata=X_train.ravel(), ydata=Y_train)
    model_custom = CustomModelWrapper(func, params)

    X_axis_reshaped = X_axis.reshape(-1, 1)
    seaborn.scatterplot(x=X_train.ravel(), y=Y_train)
    seaborn.scatterplot(x=X_test.ravel(), y=Y_test, color="white", edgecolor="black")
    seaborn.lineplot(x=X_axis, y=model_lin.predict(X_axis_reshaped), color="red")
    seaborn.lineplot(x=X_axis, y=model_custom.predict(X_axis_reshaped), color="orange")

    print("MAE_lin: " + str(mean_absolute_error(Y_test, model_lin.predict(X_test))))
    print("MAE_custom: " + str(mean_absolute_error(Y_test, model_custom.predict(X_test))))

    print("MSE_lin: " + str(mean_squared_error(Y_test, model_lin.predict(X_test))))
    print("MSE_custom: " + str(mean_squared_error(Y_test, model_custom.predict(X_test))))

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Ilość wypadków")
    plt.grid()
    plt.savefig("images/plots/" + filename)

print("Ceny wódki")
new_plot(df["wodka"], df["wypadki"], "Cena 0,5l wódki czystej 40% [zł]", "Wypadki na drogach a cena wódki", "wodka.png")
print("Wydatki województw")
new_plot(df["wydatki_woj"], df["wypadki"], "Wydatki województw na transport i łączność [tys. zł]", "Wypadki na drogach a wydatki województw na transport", "woj.png")
print("Emisja zanieczyszczeń")
new_plot(df["emisja"], df["wypadki"], "Emisja zanieczyszczeń pyłowych z zakładów szczególnie uciążliwych [ton rocznie]", "Wypadki na drogach a emisja zanieczyszczeń", "emisja.png")
print('Cena kursu kat. "B"')
new_plot(df["kurs_b"], df["wypadki"], 'Cena kursu samochodowego kat. "B" [zł]', "Wypadki na drogach a cena kursu kat. B", "kurs.png")
print("Liczba osób w wieku produkcyjnym")
new_plot(df["osob_prod"], df["wypadki"], "Liczba osób w wieku produkcyjnym (M - 18-64 lat, K - 18-59 lat)", "Wypadki na drogach a liczba osón w wieku produkcyjnym", "ludzie.png")
