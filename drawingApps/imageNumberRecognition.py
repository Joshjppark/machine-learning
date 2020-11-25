import os, pickle
import numpy as np
import tkinter as tk
from tkinter import ttk, PhotoImage
from PIL import ImageTk, Image

from MachineLearning import mnist_deepNet, optim, trainer, calc_model_accuracy

bgColor = '#505050' #equal to RGB values (80, 80 80)

def rgb2hex(rgb):
    ''' X converts to hexidecimal; %02X insures each hexidecimal
    is represented by two digits'''
    return '#%02X%02X%02X' % rgb

class Tile():

    def __init__(self, canvas, row, col):
        self.canvas = canvas
        self.pad = 5
        self.row = row * 25 + self.pad
        self.col = col * 25 + self.pad
        self.sideDim = 20
        self.colorVal = 0


        self.rect = canvas.create_rectangle(self.col, self.row, \
            self.col+self.sideDim, self.row+self.sideDim, width=0, fill='#000000')

    def clear(self):
        self.canvas.itemconfig(self.rect, fill='#000000')


    def updateTile(self, x, y):
        if (self.row < y < self.row+self.sideDim) and (self.col < x < self.col+self.sideDim):
            # print(f'Clicked on tile with coordinates {self.row, self.col}')

            self.canvas.itemconfig(self.rect, fill=rgb2hex((255, 255, 255))) #fills to white; RGB (255, 255, 255)
            self.colorVal = 255
        elif (self.row-25 < y < self.row+self.sideDim+25) and (self.col< x < self.col+self.sideDim):
            self.increaseColor()
        elif (self.row < y < self.row+self.sideDim) and (self.col-25 < x < self.col+self.sideDim+25):
            self.increaseColor()

    def increaseColor(self):
        if self.colorVal == 255:
            return
        elif self.colorVal == 0:
            self.colorVal = 128
        else:
            self.colorVal += 2

        color = (self.colorVal,self.colorVal,self.colorVal)
        self.canvas.itemconfig(self.rect, fill=rgb2hex((color)))




class numberRecognizer():

    def __init__(self, parent):
        self.parent = parent
        parent.title('Handdrawn Number Recognition DeepNet')
        parent.resizable(False, False)

        self.leftFrame = tk.Frame(self.parent, bg='')
        self.rightFrame = tk.Frame(self.parent, bg='')
        self.leftFrame.grid_propagate(False)
        self.rightFrame.grid_propagate(False)
        self.leftFrame.pack(side='left')
        self.rightFrame.pack()

        self.setupDeepNet()
        self.setupCanvas()
        self.setupRightFrame()


    def mouseClickTouch(self, event):
        x, y = event.x, event.y

        for tile in self.tileLst:
            tile.updateTile(x, y)


    def setupDeepNet(self):
        self.DeepNet = mnist_deepNet
        self.trainer = trainer

        path = "MnistData/mnist.pkl"
        # path = './MnistData/mnist.pkl'
        with open(path,'rb') as f:
            mnist = pickle.load(f)
        x_train, y_train, x_test, y_test = mnist["training_images"], mnist["training_labels"], \
            mnist["test_images"], mnist["test_labels"]
        y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)

        # make trainLabels and testLabels from y_train and y_test to use SoftmaxCrossEntrophy loss
        trainLabels = np.zeros((y_train.shape[0], 10))
        for i in range(len(y_train)):
            trainLabels[i][y_train[i]] = 1
        testLabels = np.zeros((y_test.shape[0], 10))
        for i in range(len(y_test)):
            testLabels[i][y_test[i]] = 1

        self.trainMean, self.trainStd = np.mean(x_train), np.std(x_train)
        x_train, x_test = (x_train-self.trainMean)/self.trainStd, (x_test-self.trainMean)/self.trainStd

        self.trainer.fit(x_train, trainLabels, x_test, testLabels,
                    epochs = 50,
                    eval_every = 10,
                    batch_size = 60)
        print(f'\n{calc_model_accuracy(self.DeepNet, x_test, y_test)}')
        print('\n DeepNet training complete')


    def setupCanvas(self):
        self.drawCanvas= tk.Canvas(self.leftFrame, \
            width=705, height=705, background=bgColor, bd=0, highlightthickness=0)
        self.drawCanvas.pack(padx=60, pady=95)
        

        self.tileLst = []
        for i in range(28):
            for j in range(28):
                tile = Tile(self.drawCanvas, row=i, col=j)
                self.tileLst.append(tile)


        self.drawCanvas.bind('<Button 1>', self.mouseClickTouch)
        self.drawCanvas.bind('<B1-Motion>', self.mouseClickTouch)


    def setupRightFrame(self):

        tk.Label(self.rightFrame, font=('onlyonefontexists', 30), text='Prediction', background=bgColor, foreground='white').pack(pady=30)

        imgRender = ImageTk.PhotoImage(Image.open('numberPictures/blank.jpg'))
        self.imgLabel = tk.Label(self.rightFrame, image=imgRender)
        self.imgLabel.image = imgRender
        self.imgLabel.pack()

        self.btnFrame = tk.Frame(self.rightFrame, bg='')
        self.btnFrame.pack(pady=20)

        clearBtn = tk.Label(self.btnFrame, text='Clear', font=('lorem ipsum', 30), background='#646464', \
                        foreground='white', height=2, width=8)
        predBtn = tk.Label(self.btnFrame, text='Predict', font=('lorem ipsum', 30), background='#646464', \
                        foreground='white', height=2, width=8)
    

        clearBtn.grid(row=0, column=0, padx=10)
        predBtn.grid(row=0, column=1, padx=10)
    

        clearBtn.bind('<Button-1>', self.clearCanvas)
        predBtn.bind('<Button-1>', self.predictImg)

    def clearCanvas(self, event):
        for tile in self.tileLst:
            tile.colorVal = 0
            tile.clear()

        imgRender = ImageTk.PhotoImage(Image.open('numberPictures/blank.jpg'))
        self.imgLabel.configure(image=imgRender)
        self.imgLabel.image = imgRender
        self.imgLabel.pack()


    def predictImg(self, event):
        imgArray = np.array([tile.colorVal for tile in self.tileLst])
        imgArray = (imgArray - self.trainMean)/self.trainStd
        

        self.imgPred = np.argmax(self.DeepNet.forward(imgArray, inference=True))
        print(f'Predicted Value: {self.imgPred}\nlst:{self.DeepNet.forward(imgArray, inference=True)}')

        imgPath = 'numberPictures/' + str(self.imgPred) + '.jpg'
        imgRender = ImageTk.PhotoImage(Image.open(imgPath))
        self.imgLabel.configure(image=imgRender)
        self.imgLabel.image = imgRender
        self.imgLabel.pack()


if __name__ == '__main__':
    os.chdir('/home/joshua/Desktop/python/DeepLearningFromScratch')
    root = tk.Tk()
    root.configure(background=bgColor)
    root.geometry('1500x900')
    numberRecognizer(root)
    root.mainloop()