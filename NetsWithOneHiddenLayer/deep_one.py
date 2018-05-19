"""
Модуль реализует нейронную сеть с одним скрытым слоем.
В качестве функции активации используется логистическая функция.
У нейрона выходного слоя эта функция умножается на некоторый маштабирующий множитель.
В качестве вектора признаков принимается переменная x и её степени.
"""

import numpy as np, math
import random, matplotlib.pyplot as mp


class Neuron:
    """
    Данный класс реализует модель искусственного нейрона
    и простейшие функции активации.
    """

    def __init__(self, number_input):
        self.weights = np.random.standard_normal(number_input + 1)
    
    def relu_activate(self, signals):
        res = np.dot(signals ,self.weights[1:]) + self.weights[0]
        if res > 0:
            return res
        else:
            return 0

    def linear_activate(self, signals):
        return np.dot(signals, self.weights[1:]) + self.weights[0]

    def logistic_activate(self, signals):
        res = np.dot(signals ,self.weights[1:]) + self.weights[0]
        try:
            return 1 / (1 + math.exp(-res))
        except:
            return 0


class NeuralNetwork:
    """
    Класс представляет собой нейронную сеть с одним скрытым слоем,
    где выполняется условие теоремы Колмогорова-Арнольда об апроксимации функций
    n переменных.
    """

    def __init__(self, number_arg, rate):
        self.coef = 0.01
        self.layers = []
        self.layers.append([Neuron(number_arg) for _ in range(2 * number_arg + 1 + rate)])
        self.layers.append(Neuron(2 * number_arg + 1 + rate))
        self.curv_error = []

    def activate_network(self, signals):
        """
        Метод берёт выход всей сети для заданного аргумента.
        """

        output_signals = []
        for neuron in self.layers[0]:
            output_signals.append(neuron.logistic_activate(signals))
        
        self.last_input = np.array(output_signals)
        return self.layers[1].linear_activate(self.last_input)

    def generate_data(self, len_vec):
        """Метод генерирует данные для обучающей выборки."""

        n_point = 80
        step = 0.1
        self.lenth = n_point
        self.X = np.array([[(step * i) ** j for j in range(1, len_vec + 1)] for i in range(n_point)])
        self.Y = np.array([math.sin(step * i) + 1 for i in range(n_point)])

    def get_random_sample(self):
        index = random.randint(0, self.lenth - 1)
        return self.X[index], self.Y[index]

    def SSE(self):
        result = 0.0
        for vector, tag in zip(self.X, self.Y):
            result += abs(self.activate_network(vector) - tag) / self.lenth
        return result

    def fitting(self, n_iter, len_vec):
        """Метод реализует основной цикл обучения."""

        self.generate_data(len_vec)
        n = 0

        for _ in range(n_iter):
            coef = self.coef - 0.0009 * n / n_iter
            sample, tag= self.get_random_sample()
            func = self.activate_network(sample)
            error = func - tag
            #self.curv_error.append(self.SSE())
            i = 0
            for neuron in self.layers[0]:
                new_error = error * self.layers[1].weights[i]
                activation = neuron.logistic_activate(sample)
                neuron.weights[1:] -= self.coef * new_error *  sample * activation * (1 - activation)
                neuron.weights[0] -= self.coef * new_error *  activation * (1 - activation)
                i += 1
            
            self.layers[1].weights[1:] -= coef * error * self.last_input
            self.layers[1].weights[0] -= coef * error
            n += 1
            print("Iteration is ", n, " ...")

        


if __name__ == "__main__":
    len_vec = 7
    addtional_neurons = 0
    simple_net = NeuralNetwork(len_vec, addtional_neurons)
    n_iter = 5000
    simple_net.fitting(n_iter, len_vec)
    step = 0.1
    arg_x = [i * step for i in range(160)]
    arg_y = [math.sin(e) + 1 for e in arg_x]
    out_neur = [simple_net.activate_network(np.array([row ** j for j in range(1, len_vec + 1)])) for row in arg_x]
    mp.plot(arg_x, arg_y, label = "function")
    mp.plot(arg_x, out_neur, label = "regression")
    mp.legend()
    #mp.plot(range(n_iter), simple_net.curv_error)
    mp.show()
