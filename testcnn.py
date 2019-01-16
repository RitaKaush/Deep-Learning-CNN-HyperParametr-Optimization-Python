#from PIL import Image
#import numpy, os
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.cross_validation import cross_val_score
#path_train="dataset/training_set/"
#path_test="dataset/test_set/"
##Xlist_train=[]
##Ylist_train=[]
#Xlist_test=[]
#Ylist_test=[]
##for directory in os.listdir(path_train):
##    for file in os.listdir(path_train+directory):
##       # print(path+directory+"/"+file)
##        img=Image.open(path_train+directory+"/"+file)
##        featurevector=numpy.array(img).flatten()[:50] #in my case the images dont have the
## #same dimensions, so [:50] only takes the first 50 values
##        Xlist_train.append(featurevector)
##        #print (len([Xlist]))
##        Ylist_train.append(directory)
##        #print (len([Ylist]))
#        
#for directory in os.listdir(path_test):
#    for file in os.listdir(path_test+directory):
#       
#        img=Image.open(path_test+directory+"/"+file)
#        featurevector=numpy.array(img).flatten()[:50]
#        Xlist_test.append(featurevector)
#        Ylist_test.append(directory)
#        
# 
#        
##clf_train=AdaBoostClassifier(n_estimators=100)
##scores_train = cross_val_score(clf_train, Xlist_train, Ylist_train)
#
#clf_test=AdaBoostClassifier(n_estimators=100)
#scores_test = cross_val_score(clf_test, Xlist_test, Ylist_test)
#
##print("Accuracy of Train set: %f",scores_train.mean())
#print("Accuracy of Test set: %f",scores_test.mean())


from PIL import Image
import numpy, os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import cross_val_score
path="dataset/training_set/"
Xlist=[]
Ylist=[]
for directory in os.listdir(path):
    for file in os.listdir(path+directory):
        print(path+directory+"/"+file)
        img=Image.open(path+directory+"/"+file)
        featurevector=numpy.array(img).flatten()[:4096]
        Xlist.append(featurevector)
        Ylist.append(directory)
clf=AdaBoostClassifier(n_estimators=200)
scores = cross_val_score(clf, Xlist, Ylist,cv=10,n_jobs=-1)
print("Accuracy of Training set: ",scores.mean())