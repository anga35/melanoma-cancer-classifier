import os
import pickle
import cv2
import numpy as np

with open('C:/Dev/python_demos/model.pkl','rb') as file:
    model=pickle.load(file=file)

test_dir='C:/Users/Dayod/Downloads/melanoma_cancer_dataset/test/malignant/'
img_set=[]
i=0
for file in os.listdir(test_dir):

    filename=os.path.join(test_dir,file)
    img=cv2.imread(filename=filename)
    img=cv2.resize(img,(100,100))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.normalize(img,None,alpha=0,beta=1,
    norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    img=img.flatten()
    img_set.append(img)
    if(i==40):
        break

print('predicting')
img_set=np.asarray(img_set)
res=model.predict(img_set)
print(res)