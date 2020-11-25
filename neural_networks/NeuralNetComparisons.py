import models
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

boston = load_boston()
data = boston.data
target = boston.target

s = StandardScaler()
data = s.fit_transform(data)

linear_regression = models.NeuralNetwork(
        layers=[models.Dense(neurons=1, activation=models.Linear())],
        loss=models.MeanSquaredError(),
        seed=20190501
        )

neural_network = models.NeuralNetwork(
        layers=[
            models.Dense(neurons=13, activation=models.Sigmoid()),
            models.Dense(neurons=1, activation=models.Linear())
            ],
        loss=models.MeanSquaredError(),
        seed=20190501
        )

deep_neural_network = models.NeuralNetwork(
    layers=[
        models.Dense(neurons=13, activation=models.Sigmoid()),
        models.Dense(neurons=13, activation=models.Sigmoid()),
        models.Dense(neurons=1, activation=models.Linear())
    ],
    loss=models.MeanSquaredError(),
    seed=20190501
)



x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.3, random_state=80718)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)



linear_trainer = models.Trainer(
    linear_regression,
    models.SGD(init_learning_rate=0.01)
    )

print()
print("Linear Regression")
print()
linear_trainer.fit(
    x_train, y_train, x_test, y_test,
    epochs = 50,
    eval_every = 10,
    batch_size = 32,
    seed = 20190501,
    restart=True)
print()
models.eval_regression_model(linear_regression, x_test, y_test)

# x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=.3, random_state=80718)
# y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

neural_trainer = models.Trainer(
    neural_network,
    models.SGD(init_learning_rate=0.01)
    )

print()
print("Neural Network")
print()
neural_trainer.fit(
    x_train, y_train, x_test, y_test,
    epochs = 50,
    eval_every = 10,
    batch_size = 32,
    seed = 20190501,
    restart=True)


print()
models.eval_regression_model(neural_network, x_test, y_test)


deep_trainer = models.Trainer(
    deep_neural_network,
    models.SGD(init_learning_rate=0.01)
    )

print()
print("Deep Learning")
print()
deep_trainer.fit(
    x_train, y_train, x_test, y_test,
    epochs = 50,
    eval_every = 10,
    batch_size = 32,
    seed = 20190501,
    restart = True)
print()
models.eval_regression_model(deep_neural_network, x_test, y_test)