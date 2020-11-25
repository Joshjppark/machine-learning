import numpy as np

from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.lang import Builder

from Models.utils import *
from Models.base import *
from Models.activations import *
from Models.layers import *
from Models.losses import *
from Models.network import *
from Models.optimizer import *
from Models.parameters import *
from Models.trainer import *

import pickle, os

def load():
    os.chdir('Mnist_Data')
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    os.chdir('../')
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]


from kivy.config import Config
Config.set('graphics', 'resizeable', 'False')
white = (255, 255, 255)
black = (0, 0, 0)

from kivy.core.window import Window
Window.size = (1200, 720)


class Tile(Widget):

    def __init__(self, row, column):
        super().__init__()
        self.row = 20 + row * 50
        self.column = 50 + column * 50
        self.index = (row, column)
        self.side_dimen = 40
        self.color = 0


        with self.canvas:
            Color(0, 0, 0)

            # Add a rectangle
            self.rect = Rectangle(pos=(self.column, self.row), size=(self.side_dimen, self.side_dimen))
    # def on_touch_down(self, touch):
    #     half_side = self.side_dimen / 2

    #     center = (self.column + half_side, self.row + half_side)

    #     if (touch.x - 50 < center[0] < touch.x + 50) and \
    #         (touch.y - 50 < center[1] < touch.y + 50):
    #         self.addGray()


    # def on_touch_move(self, touch):
    #     half_side = self.side_dimen / 2

    #     center = (self.column + half_side, self.row + half_side)

    #     if (touch.x - 30 < center[0] < touch.x + 30) and \
    #         (touch.y - 30 < center[1] < touch.y + 30):
    #         self.addGray()

    def on_touch_down(self, touch):
        if (self.row < touch.y < self.row + self.side_dimen) and \
            (self.column < touch.x < self.column + self.side_dimen):
            self.changeWhite()
        elif (self.row < touch.y+50 < self.row + self.side_dimen) and \
            (self.column < touch.x < self.column + self.side_dimen):
            self.addGray()
        elif (self.row < touch.y-50 < self.row + self.side_dimen) and \
            (self.column < touch.x < self.column + self.side_dimen):
            self.addGray()
        elif (self.row < touch.y < self.row + self.side_dimen) and \
            (self.column < touch.x+50 < self.column + self.side_dimen):
            self.addGray()
        elif (self.row < touch.y < self.row + self.side_dimen) and \
            (self.column < touch.x-50 < self.column + self.side_dimen):
            self.addGray()


    def on_touch_move(self, touch):
        if (self.row < touch.y < self.row + self.side_dimen) and \
            (self.column < touch.x < self.column + self.side_dimen):
            self.changeWhite()
        elif (self.row < touch.y+50 < self.row + self.side_dimen) and \
            (self.column < touch.x < self.column + self.side_dimen):
            self.addGray()
        elif (self.row < touch.y-50 < self.row + self.side_dimen) and \
            (self.column < touch.x < self.column + self.side_dimen):
            self.addGray()
        elif (self.row < touch.y < self.row + self.side_dimen) and \
            (self.column < touch.x+50 < self.column + self.side_dimen):
            self.addGray()
        elif (self.row < touch.y < self.row + self.side_dimen) and \
            (self.column < touch.x-50 < self.column + self.side_dimen):
            self.addGray()

    def changeWhite(self):
        self.color = 255
        with self.canvas:
            Color(1, 1, 1)
            Rectangle(pos=(self.column, self.row), size=(self.side_dimen, self.side_dimen))

    def changeBlack(self):
        self.color = 0
        with self.canvas:
            Color(0, 0, 0)
            Rectangle(pos=(self.column, self.row), size=(self.side_dimen, self.side_dimen))

    def addGray(self):
        self.color += 100
        colorvalue = self.color * .0039
        with self.canvas:
            Color(colorvalue, colorvalue, colorvalue)
            # Color(.2, .2, .2)
            Rectangle(pos=(self.column, self.row), size=(self.side_dimen, self.side_dimen))


class gridBoxPicture(Widget):

    def __init__(self, **kwargs):
        super(gridBoxPicture, self).__init__(**kwargs)
        self.Tilelst = []

        for i in range(28):
            for j in range(28):
                self.Tilelst.append(Tile(row=i, column=j))

        self.layout = GridLayout(cols=28, padding=20, spacing=5)
        for i in range(784):
                self.layout.add_widget(self.Tilelst[i])



class drawingApp(App):

    def build(self):

        root = BoxLayout()

        # Define gridbox layout and grixBoxPicture object
        self.gridbox = gridBoxPicture(size_hint=(.6, 1), pos=(0, 0))
        self.gridboxLayout = self.gridbox.layout
        
        # Layout for the right hand side
        self.rightBoxLayout = BoxLayout(orientation='vertical', size_hint=(.5, 1), padding=50, spacing=50)

        self.source = 'numberPictures/blank.jpg'
        self.rightBox()


        # Add widgets to the root
        root.add_widget(self.gridboxLayout)
        root.add_widget(self.rightBoxLayout)

        self.train_model()

        return root


    def rightBox(self):

        self.rightBoxLayout.clear_widgets()
        # Add labels to the rightbox
        self.rightBoxLayout.add_widget(Label(text='Your Number:', font_size=80, size_hint=(1, .3)))
        self.rightBoxLayout.add_widget(Label(size_hint=(1, .1)))
        self.rightBoxLayout.add_widget(Image(source = self.source))


        # Button layout
        buttonLayout = BoxLayout(orientation = 'horizontal', padding=10, spacing=20)
        clearbtn = Button(text='Clear', size_hint=(1, .7))
        predbtn = Button(text='Predict', size_hint=(1, .7))
        clearbtn.bind(on_release=self.clear_grid)
        predbtn.bind(on_release=self.pred_grid)
        buttonLayout.add_widget(clearbtn)
        buttonLayout.add_widget(predbtn)


        self.rightBoxLayout.add_widget(buttonLayout)

    def print_image(self):
        self.rightBoxLayout.add_widget(Image(source = self.source))

    def train_model(self):
        deep_model = NeuralNetwork(
            layers=[Dense(neurons=178, activation=Tanh(), weight_init='glorot', dropout=.8),
                    Dense(neurons=46, activation=Tanh(), weight_init='glorot', dropout=.8),
                    Dense(neurons=10, activation=Linear(), weight_init='glorot')],
            loss=SoftMaxCrossEntrophy(),
            seed=20190119)

        self.deep_trainer = Trainer(deep_model, SGDMomentum(lr=.2, momentum=.9, final_lr=.05, decay_type='exponential'))
        self.deep_trainer.fit(x_train, train_labels, x_test, test_labels,
                epochs=3,
                eval_every=10,
                batch_size=60,
                seed=20190119)
        print()
        eval_model_accuracy(deep_model, x_test, y_test)

    def clear_grid(self, obj):
        
        for Tile in self.gridbox.Tilelst:
            Tile.changeBlack()

    def pred_grid(self, obj):
        colorArray = []
        for Tile in self.gridbox.Tilelst:
            colorArray.append(Tile.color)

        colorArray = np.clip(colorArray, 0, 255)
        npColorArray = np.array(colorArray).reshape(28, 28)
        npColorArray = np.flip(npColorArray, 0)
        npColorArray = np.array(npColorArray).reshape(784,)

        self.predict_num(npColorArray)

    def predict_num(self, npColorArray):
        normArray = (npColorArray - x_test_mean) / x_test_std
        arrayResult = self.deep_trainer.net.forward(normArray, inference=True)
        numResult = np.argmax(arrayResult)
        print(numResult)

        self.source = f"numberPictures/{str(numResult)}.jpg"
        self.rightBox()
        
Builder.load_string('''
<BoxLayout>:
    canvas.before:
        Color:
            rgba: .3, .3, .3, 1
        Rectangle:
            pos: self.pos
            size: self.size
<Label>:
    size: (2, 2)
<Image>:
    size_hint_y: None
    height: dp(400)
    width: dp(400)
''')

x_train, y_train, x_test, y_test = load()
train_labels, test_labels = one_hot_encoding(y_train, y_test)
x_train = (x_train - np.mean(x_train)) / np.std(x_train)

x_test_mean, x_test_std = np.mean(x_test), np.std(x_test)
x_test = (x_test - x_test_mean) / x_test_std

if __name__ == '__main__':
    drawingApp().run()