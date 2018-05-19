"""
Модуль реализует нейронную сеть с одним скрытым слоем.
В качестве функции активации используется логистическая функция.
У нейрона выходного слоя эта функция умножается на некоторый маштабирующий множитель.
В качестве вектора признаков принимается последовательность Yn, Yn+1 и т.д до Yn+k 
для предсказания Yn+k+1.
"""

import numpy as np, math, sys
import _pickle  as pickle
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
        self.scale = 40
        self.width = number_arg
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
        return self.scale * self.layers[1].linear_activate(self.last_input)

    def generate_data(self):
        """Метод генерирует данные для обучающей выборки."""

        step = 0.01
        n_point = 500
        
        self.X = np.array([step * i for i in range(n_point)])
        self.Y = np.array([abs(math.sin(step * i )) for i in range(n_point)])


    def get_random_sample(self):
        
        lenth = len(self.Y)
        index = random.randint(0, lenth - 1 - width)
        return self.Y[index : index + self.width], self.Y[index + self.width]

    def SSE(self):
        lenth = len(self.Y)
        result = 0.0
        for vector, tag in zip(self.X, self.Y):
            result += abs(self.activate_network(vector) - tag)
        return result

    def fitting(self, n_iter):
        """Метод реализует основной цикл обучения."""

        self.generate_data()
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
    width = 2
    simple_net = NeuralNetwork(width, 0)
    n_iter = 50000
    points = 200
    simple_net.fitting(n_iter)
    arg_x = [ i * 0.1 for i in range(points)]
    arg_y = [abs(math.sin(e)) for e in arg_x]
    out_neur = [simple_net.activate_network(np.array(arg_y[i:i + width])) for i in range(points - width)]
    mp.plot(arg_x[:points-width], arg_y[:points-width], label = "function")
    mp.plot(arg_x[:points-width], out_neur, label = "regrssion")
    mp.legend()
    mp.show()
    #mp.plot(range(n_iter), simple_net.curv_error)
    #mp.show()
    if len(sys.argv) > 1 and sys.argv[1] == "yes":
    	saver = open("neuron.pkl", "wb")
    	pickle.dump(simple_net, saver)
    	saver.close()
