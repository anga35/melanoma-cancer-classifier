import os
import numpy as np
import cv2
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import StratifiedShuffleSplit,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

base_dir='C:/Users/Dayod/Downloads/melanoma_cancer_dataset/train/'
images=[]
labels=[]
categories=['benign','malignant']

for(category_indx,category) in enumerate(categories):
    for file in os.listdir(os.path.join(base_dir,category)):
        img_dir=os.path.join(base_dir,f'{category}/',file)
        image=cv2.imread(img_dir)
        image=cv2.resize(image,(100,100))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image=cv2.normalize(image,None,alpha=0,beta=1,
        norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        image=image.flatten()
        images.append(image)
        labels.append(category_indx)


image_dataset= np.asarray(images)
label_dataset=np.asarray(labels)

ss=StratifiedShuffleSplit(n_splits=1,test_size=0.01,random_state=0)
for train_index,test_index in ss.split(image_dataset,label_dataset):
    x_shuffled,y_shuffled=image_dataset[train_index],label_dataset[train_index]
    x_test,y_test=image_dataset[test_index],label_dataset[test_index]

classifier=SVC()

# hyper_parameters=[{'gamma':[0.01,0.001,0.0001],'C':[1,10,100,1000]}]

# Set an early stopping condition
# threshold = 0.9


classifier.fit(x_shuffled, y_shuffled)

#performance
best_estimator=classifier 
'''random_search.best_estimator_'''
y_prediction=best_estimator.predict(x_test)
score=accuracy_score(y_true=y_test,y_pred=y_prediction)

# if score > threshold:
#     pickle.dump(best_estimator,open('./model.pkl','wb'))

print(f'The accuracy score is {score}')

# image=cv2.imread('my_pic.jpg')

# print(image.flatten())


# img=imread('my_pic.jpg')
# print(img.flatten())
