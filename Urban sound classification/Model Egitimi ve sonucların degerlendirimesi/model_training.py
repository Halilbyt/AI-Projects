from logging.config import valid_ident
from tabnanny import verbose
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sb
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf



class Model_DNN(Sequential):
    def __init__(self,train_path,test_path,valid_path):
        self.train_path = train_path
        self.test_path = test_path
        self.valid_path = valid_path
        super.__init__(self)
        self.sequential = Sequential

    def __str__(self,model):
        return model.summary()

    def dataPreprocessing(self,batch_size,target_size, data_agumentation=False):

        # Eğitim verisini çeşitlendirmek için
        if(data_agumentation == True):
             
            train_data_gen = ImageDataGenerator(
                        rescale=1./255,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        zoom_range=0.2,
                        horizontal_flip=True,
                        fill_mode="nearest",  
                        target_size = target_size,
                        shuffle=False,
                        batch_size=batch_size,
                        class_mode="categorical")

            test_data_gen = ImageDataGenerator(rescale=1./255)
            
            valid_data_gen = ImageDataGenerator(rescale=1./255)

        else:

            # Resim dosyalarını 0-1 arasında reniden ölçeklendir.
            train_data_gen = ImageDataGenerator(rescale=1./255)
            
            test_data_gen = ImageDataGenerator(rescale=1./255)
            
            valid_data_gen = ImageDataGenerator(rescale=1./255)
   

        # Dosya pathlerini al
        
        train_path = self.train_path
        test_path = self.test_path
        valid_path = self.valid_path

        # Dosyaların boyut ve her epoch'da alınacak veri boyutunu belirle.
        
        batch_size = target_size
        target_size = batch_size
        
        # train,test ve validation veri generatörlerini çalıştır.
  
        train = train_data_gen.flow_from_directory(
                train_path,
                target_size = target_size,
                shuffle=False,
                batch_size=batch_size,
                class_mode="categorical")

        test = test_data_gen.flow_from_directory(
                test_path,
                shuffle=False,
                target_size = target_size,
                batch_size=batch_size,
                class_mode="categorical")

        valid = valid_data_gen.flow_from_directory(
                valid_path,
                shuffle=False,
                target_size = target_size,
                batch_size=batch_size,
                class_mode="categorical")
        
        return train, test, valid     

    def checkandSetGPU(self):

        # GPU kontrolu ve hafıza limitini düzenlemek için 
        
        gpu = tf.config.list_physical_devices('GPU')
        
        if gpu:
            try:
                tf.config.experimental.set_visible_devices(gpu[0],'GPU')
                tf.config.experimental.set_virtual_device_configuration(gpu[0],[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1500)])
            except RuntimeError as e:
                return print(e)
        else:
            return print("GPU bulunamadı")

    def modelCallbacks(self,patience):
        
        # Modelin overfitting'e gitmemesi ve gereksiz yere eğitimi sürdürmemesi için 
        
        earlyStop = EarlyStopping(
            monitor = "val_loss",
            patience=patience,
            verbose=1,
            mode = "min"
        )

        # Otomatik ağ kaydı için

        model_check=ModelCheckpoint(
            "saved_model",
            monitor = "val_accuracy",
            mode="max",
            verbose=1,
            save_weight_only=False
        )
        return earlyStop, model_check 

    def creatingModel(self,input_shape=(96,96,3),strides=(1,1),activation="relu",output_activation="softmax",kernel=(3,3),numberofnode=32):
            
            model = self.sequential()
            model.add(Conv2D(numberofnode,kernel,input_shape=input_shape,strides=strides,activation=activation))
            model.add(MaxPooling2D((2,2)))
            model.add(Conv2D(numberofnode*2,kernel,strides=strides,activation=activation))
            model.add(MaxPooling2D((2,2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(numberofnode*3,kernel,strides=(1,1),activation="relu"))
            model.add(MaxPooling2D((2,2)))
            model.add(Dropout(0.5))
            model.add(Conv2D(numberofnode*2,kernel,strides=(1,1),activation="relu"))
            model.add(MaxPooling2D((2,2)))
            model.add(Dropout(0.5))
            model.add(Flatten())
            model.add(Dense(512,activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(256,activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(10,activation=output_activation))

            return model
        
    def trainingModel(self,train,valid,optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"],earlyStopping=None):
        
        model = self.creatingModel()
        
        model.compile(optimizer,loss,metrics)
        
        model.fit(train,epochs=250,
                    validation_data=valid,
                    verbose=1,
                    batch_size=32,
                    callbacks=[earlyStopping])
        
        if(float(model.history.history["accuracy"][-1])>0.9):
            model.save("best_model")
            return model,model.evaluate(test)
        else:
            return print("Accuracy %90'ın altında")

    def plottingResults(self,model):
        # Doğruluk ve kayıp değerlerini hem train hem de validation için elde ediyoruz
        acc = model.history.history["accuracy"]
        val_acc = model.history.history["val_accuracy"]
        
        epochs = len(acc)+1
        
        loss = model.history.history["loss"]
        val_loss = model.history.history["val_loss"]
        
        plt.figure(figsize=(12,8))
        
        plt.subplots(2,1,1)
        plt.plot(epochs,acc,color="blue",label="Eğitim Başarım Oranı")
        plt.plot(epochs,val_acc,color="orrange",label="Validation Başarım Oranı" )
        plt.xlabel("Epochs")
        plt.ylabel("Doğruluk")
        plt.lagend()

        plt.subplots(2,1,2)
        plt.plot(epochs,loss,color="blue",label="Eğitim Kayıp Oranı")
        plt.plot(epochs,val_loss,color="orrange",label="Validation Kayıp Oranı")
        plt.xlabel("Epochs")
        plt.ylabel("Kayıp")
        plt.legend()
        plt.show()
        


if __name__ =="__main__":

    train_path = r"C:\Users\D4rkS\Desktop\python evr\UrbanDataset_tr_tst_val\train"
    test_path = r"C:\Users\D4rkS\Desktop\python evr\UrbanDataset_tr_tst_val\test"
    valid_path = "C:\Users\D4rkS\Desktop\python evr\UrbanDataset_tr_tst_val\val"

    DNN_object = Model_DNN(train_path,test_path,valid_path)

    train,test,valid = DNN_object.dataPreprocessing(32,(64,64,3),False)

    DNN_object.creatingModel(input_shape=(64,64,3),strides=(1,1),activation="relu",output_activation="softmax",kernel=(3,3),numberofnode=32)
    
    model,test_result = DNN_object.trainingModel(train,valid,optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"],earlyStopping=None)
    # Sonuç çizimlerinin gösterimi için
    DNN_object.plottingResults(model)