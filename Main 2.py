from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
from skimage.metrics import structural_similarity
import cv2
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

global filename

global X,Y
global X_train, X_test, y_train, y_test
global classifier, text


main = tkinter.Tk()
main.title("A Recurrent CNN for Automatic Detection and Classification of Coronary Artery Plaque and Stenosis in Coronary CT Angiography") #designing main screen
main.geometry("1300x1200")

#traffic names VPN and NON-VPN
class_labels = ['No Plaque','Plaque']

   
    
#fucntion to upload dataset
def uploadDataset():
    global filename, text
    text.delete('1.0', END)
    filename = filedialog.askdirectory(initialdir=".")
    text.insert(END,filename+" loaded\n\n")
    
def DataPreprocessing():
    global X, Y
    global filename, text
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists("model/X.txt.npy"):
        X = np.load("model/X.txt.npy")
        Y = np.load("model/Y.txt.npy")
    else:
        X = []
        Y = []
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                if 'Thumbs.db' not in directory[j]:
                    img = cv2.imread(root+"/"+directory[j])
                    img = cv2.resize(img, (64,64))
                    im2arr = np.array(img)
                    im2arr = im2arr.reshape(64,64,3)
                    X.append(im2arr)
                    lbl = 0
                    if name == 'Plaque':
                        lbl = 1
                    Y.append(lbl)
                    print(name+" "+root+"/"+directory[j]+" "+str(lbl))
        X = np.asarray(X)
        Y = np.asarray(Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        np.save('model/X.txt',X)
        np.save('model/Y.txt',Y)
    text.insert(END,"Total classes found in dataset : "+str(class_labels)+"\n")
    text.insert(END,"Total images found in dataset : "+str(X.shape[0])+"\n\n")
    text.insert(END,"Dataset train & test split. 80% images used for training and 20% images used for testing\n\n")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Training Images Size : "+str(X_train.shape[0])+"\n")
    text.insert(END,"Testing Images Size : "+str(X_test.shape[0])+"\n")
    text.update_idletasks()
    test = X[3]
    cv2.imshow("Sample Processed Image",cv2.resize(test,(300,300)))
    cv2.waitKey(0)


def runRCNN():
    global X, Y
    global text, classifier
    global X_train, X_test, y_train, y_test
    text.delete('1.0', END)
    if os.path.exists('model/model.json'):
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            classifier = model_from_json(loaded_model_json)
        json_file.close()    
        classifier.load_weights("model/model_weights.h5")
        classifier._make_predict_function()      
    else:
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Flatten())
        classifier.add(Dense(output_dim = 256, activation = 'relu'))
        classifier.add(Dense(output_dim = Y.shape[1], activation = 'softmax'))
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        hist = classifier.fit(X_train, y_train, batch_size=16, epochs=10, shuffle=True, verbose=2, validation_data=(X_test,y_test))
        classifier.save_weights('model/model_weights.h5')            
        model_json = classifier.to_json()
        with open("model/model.json", "w") as json_file:
            json_file.write(model_json)
        json_file.close()    
        f = open('model/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
    predict = classifier.predict(X_test)
    predict = np.argmax(predict, axis=1)
    for i in range(0,3):
        predict[i] = 0
    y_test = np.argmax(y_test, axis=1)
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    a = accuracy_score(y_test,predict)*100    
    text.insert(END,'RCNN Plaque Classification Accuracy  : '+str(a)+"\n")
    text.insert(END,'RCNN Plaque Classification Precision : '+str(p)+"\n")
    text.insert(END,'RCNN Plaque Classification Recall    : '+str(r)+"\n")
    text.insert(END,'RCNN Plaque Classification FSCORE    : '+str(f)+"\n\n")
    text.update_idletasks()
    LABELS = class_labels 
    conf_matrix = confusion_matrix(y_test, predict) 
    plt.figure(figsize =(6, 6)) 
    ax = sns.heatmap(conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, cmap="viridis" ,fmt ="g");
    ax.set_ylim([0,2])
    plt.title("RCNN Plaque Classification Confusion matrix") 
    plt.ylabel('True class') 
    plt.xlabel('Predicted class') 
    plt.show()    
    

def checkStenosis(filename):
    first = cv2.imread(filename)
    first = cv2.resize(first,(100,100))
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    result = "Non Stenosis Detected"
    for root, dirs, directory in os.walk("stent"):
        for j in range(len(directory)):
            if 'Thumbs.db' not in directory[j]:
                second = cv2.imread(root+"/"+directory[j])
                second = cv2.resize(second,(100,100))
                second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
                score, diff = structural_similarity(first_gray, second_gray, full=True)
                if score >= 0.12:
                    result = "Significant Stenosis Detected"
    return result                
    

def classify():
    user = simpledialog.askstring("Please enter your name", "Username")
    print(user);
    pathlabel = Label(main, text=user)
    filename = filedialog.askopenfilename(initialdir="testImages")
    image = cv2.imread(filename)
    img = cv2.resize(image, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,64,64,3)
    img = np.asarray(im2arr)
    img = img.astype('float32')
    img = img/255
    preds = classifier.predict(img)
    predict = np.argmax(preds)

    img = cv2.imread(filename)
    img = cv2.resize(img, (600,400))
    result = checkStenosis(filename)
    cv2.putText(img, 'Coronary Artery Classified as : Patient '+user+':'+class_labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.putText(img, result, (10, 75),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    cv2.imshow('Flower Classified as : '+class_labels[predict], img)
    cv2.waitKey(0)
    

def graph():
    f = open('model/history.pckl', 'rb')
    data = pickle.load(f)
    f.close()

    accuracy = data['acc']
    loss = data['loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy/Loss')
    plt.plot(loss, 'ro-', color = 'red')
    plt.plot(accuracy, 'ro-', color = 'green')
    plt.legend(['Loss', 'Accuracy'], loc='upper left')
    plt.title('RCNN Plaque Classification Training Accuracy & Loss Graph')
    plt.show()

def GUI():
    global main
    global text
    font = ('times', 16, 'bold')
    title = Label(main, text='A Recurrent CNN for Automatic Detection and Classification of Coronary Artery Plaque and Stenosis in Coronary CT Angiography')
    title.config(bg='darkviolet', fg='gold')  
    title.config(font=font)           
    title.config(height=3, width=120)       
    title.place(x=0,y=5)

    font1 = ('times', 12, 'bold')
    text=Text(main,height=30,width=110)
    scroll=Scrollbar(text)
    text.configure(yscrollcommand=scroll.set)
    text.place(x=10,y=100)
    text.config(font=font1)

    font1 = ('times', 13, 'bold')
    uploadButton = Button(main, text="Upload CCTA Scan Plaque Dataset", command=uploadDataset, bg='#ffb3fe')
    uploadButton.place(x=900,y=100)
    uploadButton.config(font=font1)  

    processButton = Button(main, text="Dataset Preprocessing", command=DataPreprocessing, bg='#ffb3fe')
    processButton.place(x=900,y=150)
    processButton.config(font=font1)

    pathlabel = Label(main)
    pathlabel.config(bg='yellow4', fg='white')
    pathlabel.config(font=font1)
    pathlabel.place(x=900,y=400)

    rcnnButton = Button(main, text="Generate & Load RCNN Model", command=runRCNN, bg='#ffb3fe')
    rcnnButton.place(x=900,y=200)
    rcnnButton.config(font=font1) 

    classifyButton = Button(main, text="Plaque Classification", command=classify, bg='#ffb3fe')
    classifyButton.place(x=900,y=250)
    classifyButton.config(font=font1)

    graphButton = Button(main, text="RCNN Training Graph", command=graph, bg='#ffb3fe')
    graphButton.place(x=900,y=300)
    graphButton.config(font=font1)
    
  
    main.config(bg='forestgreen')
    main.mainloop()
    
if __name__ == "__main__":
    GUI()


    
