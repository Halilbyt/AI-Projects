from re import sub
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import librosa.display
import librosa 
import splitfolders as sf

class dataPrep:
    
    def __init__(self,csv_path,data_path,newPath):        
        self.df = pd.read_csv(csv_path)
        self.data_path = data_path
        self.newPath =  newPath 
        self.createdFilePath, self.msg = self.creatFolder(newPath)       
        
    def __len__(self):
        return len(self.df)    

    def __getitem__(self,index):
         return self.getDataInfo.iloc[index]
    
    def getDataInfo(self):
        # sadece dosya ismi, class no ve dosya adı bilgilerini al
        dataInfo = self.df.drop(["start","end","fsID","salience","classID"],axis=1)
        return dataInfo

    def creatFolder(self):
        
        # Dosya yolunu al
        folder_name = self.newPath
        
        # Dosyayı oluştur
        try:
            os.makedirs(folder_name)
        except:
            raise Exception("Dosya yolu ve ismini duzgun giriniz.")
            
        
        msg = print(f"{folder_name} isimli dosya oluşturuldu")
        createdFilePath = self.newPath+"\\"+folder_name
        # Oluşturulan dosya ve yolunu geri döndür.
        return createdFilePath, msg

    def prepareAudioFileByClassName(self):
        
        # Auido dosyalarını 10 tane class ismini içeren dosyalara yeniden konumlandırmak için
        
        for i in range(len(self.df)):

            #csv dosyasından sırayla dosya,klasör ve class isimlerinin okunması
            
            file_name = self.getDataInfo.loc[i][0]
            fold_name = self.getDataInfo[i][1]
            class_name = self.getDataInfo[i][2] 

            for folder in os.listdir(self.data_path):                
               
                #Mevcut indexe ait audio dosyasının dosya ismi dizindeki klasör isimlerinin herhangi biriyle eşleşmesi durumu
                
                if(("fold"+str(fold_name))==(folder)):
                    
                    # Mevcut ses dosyasının ismini bulana kadar dizindeki klosörleri dolaş eşleşmeyi yakaladığında  o dosya içerisinde diğer bir döngüye gir
                    
                    for audio_files in os.listdir(self.audioPath+"fold"+f"{fold_name}"):
                        
                        # İlgili dosya içerisinde başta belirtilen ses dosyasının ismi ile eşleşeni bul
                        
                        if(file_name==audio_files):

                            # Daha Önce oluşturduğumuz klasörün içerisinde bu ses dosyasının class isminde bir klasör var mı 
                            # yoksa klasörü oluştur ve devam et

                            if(os.path.exists(self.createdFilePath+f"{class_name}")==True):
                                #Fonksiyonu ile mfcc spectrogram resimleri png formatında ilgili closerlere kaydedilir.
                                self.createMfcc(fold_name,file_name,class_name)
                                                    
                               
                            else:
                                os.makedirs(self.createdFilePath+f"{class_name}")
                                self.createMfcc(fold_name,file_name,class_name)                                

    def createMfcc(self,fold_name,file_name,class_name):        
         
         #Ses dosyalarının mfcc spectrogramlarının elde edilmesi 
         
         sound,sr = librosa.load(self.createdFilePath+f"fold{fold_name}\\{file_name}",sr=sr)
         mfcc_spect = librosa.feature.melspectrogram(sound,sr=sr)
         
         # Mfcc spectrogramlarının  matplotlib aracılığıyla  kaydedilmesi süreci
         fig,ax = plt.subplots(1,figsize=(12,8))
         mels_spec = librosa.amplitude_to_db(mfcc_spect,ref=np.max)
         mfcc_figure = librosa.display.specshow(mels_spec,sr=sr,ax=ax,y_axis="linear")

         #Kayıt sırasında ekrana görsellerin yansımaması ve işlem ağırlığını hafifletmek için 
         
         ax.axes.get_xaxis().set_visible(False)
         ax.axes.get_yaxis().set_visible(False)
         ax.set_frame_on(False)
         ax.set_xlabel(None)
         ax.set_ylabel(None)
         #Show komutunu yazmadan siyah ekran kaydediyor bilginize..
         plt.show()
         fig.savefig(self.createdFilePath+f"{class_name}"+"\\"+f"{file_name}.png")
         plt.close(fig)
    
    def splitFolders(self,path,output_path):
        # Buradaki islem 10 class ismine göre oluşturduğumuz klasörleri train,
        # test ve validation olmak üzere aşağıdaki oranda ayırıp farklı bir klasöre bunları kaydeder. 
        sf.ratio("path",
                output="output_path",
                seed=42,
                ratio=(.7, .2, .1),
                group_prefix=None) 
        """
                output/
            train/
                class1/
                    img1.jpg
                    ...
                class2/
                    imga.jpg
                    ...
            val/
                class1/
                    img2.jpg
                    ...
                class2/
                    imgb.jpg
                    ...
            test/
                class1/
                    img3.jpg
                    ...
                class2/
                    imgc.jpg
                    ...
        """

        
if __name__ == "__main__":
    
    dataPathCsv = r"C:\Users\D4rkS\Desktop\UrbanSoundClassification\urbanSound8K\metadata"
    audioPath = r"C:\Users\D4rkS\Desktop\UrbanSoundClassification\urbanSound8K\audio"
    newpath   = r"C:\Users\D4rkS\Desktop\UrbanSoundClassification\urbanSound8K"
    
    # classın objesini oluştur gerekli parametreleri gir
    my_object = dataPrep(dataPathCsv,audioPath,newpath)
    # verinin ilk elemanınını ekrana yazdır
    print(my_object[0])
    # csv dosyasının verilerini dataframe formatında geri döndürür
    my_object.getDataInfo()
    # dosyaları oluştur.
    my_object.prepareAudioFileByClassName()



