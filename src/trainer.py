#!/usr/bin/python

# import numpy as np
# import glob
# # import pcl
# # import csv

# # with open('../build/features.csv', 'rb') as f:
# #     reader = csv.reader(f)
# #     for row in reader:
# #         print row

# NaN = float('nan')
# features = np.genfromtxt('../build/features.csv',delimiter=',') #,dtype=float
# print len(features),features[0]


# cleaned_list = [x for x in features if ~np.isnan(x)] 
# 	# features.remove(NaN)
# print len(cleaned_list), cleaned_list[0]


import glob
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.tree
import pcl
import colorsys
import cPickle as pickle
import sklearn.svm
import re
import matplotlib.pyplot as plt

labels = ["Apfelsaft","BigKetchupBottle","BlackPringles","DoppelkeksBiscuit","KaffeeBox",
                                "Maggi","Messmer","MuscleBox","Orangensaft","RedBull","RedCup","SmallKetchupBottle",
                                "Sponge","YellowPringles"]
class ObjectClassifier:
    """
    Defines an SVM classifier with the mean and standard deviation of
    the features, and a label encoder

    """
    def __init__(self, classifier, label_encoder, mean, std):
        self.classifier = classifier
        self.label_encoder = label_encoder
        self.mean = mean
        self.std = std

    def save(self, classifier_name, label_encoder_name):
        with open(classifier_name, 'wb') as f:
            pickle.dump(self.classifier, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(label_encoder_name, 'wb') as f:
            pickle.dump([self.label_encoder, self.mean, self.std], f, protocol=pickle.HIGHEST_PROTOCOL)

    def classify(self, feature_vector):
        feature_vector -= np.array(self.mean)
        feature_vector /= self.std
        probabilities = self.classifier.predict_proba(feature_vector)[0]
        max_index = np.argmax(probabilities)
        cls = self.classifier.classes_[max_index]
        return self.label_encoder.inverse_transform(cls), probabilities[max_index]

    @classmethod
    def load(cls, classifier_name, label_encoder_name):
        with open(classifier_name, 'rb') as f:
            classifier = pickle.load(f)
        with open(label_encoder_name, 'rb') as f:
            [label_encoder, mean, std] = pickle.load(f)
        return ObjectClassifier(classifier, label_encoder, mean, std)


class Trainer:

    def __init__(self, data_folder):
        self.data_folder = data_folder

    def train(self, objects='all'):
        objects_to_train = []
        object_directories = np.array(glob.glob(self.data_folder + '/*'))
        if objects == 'all':
            for obj_dir in object_directories:
                object_name = obj_dir.split('/')[-1]
                objects_to_train.append(str(object_name))
        else:
            objects_to_train = objects

        print "Training classifer for objects: ", objects_to_train

        n = 0
        feature_pool = np.empty([0, 0])
        label_pool = []
        for obj in objects_to_train:
            files = np.array(glob.glob(self.data_folder + '/' + obj + '/*'))
            for f in files:
                features = np.ravel(np.genfromtxt(f,delimiter=','))
                if n < 1 :
					len_of_file = len(features)
					feature_pool = np.array(features)
					label_pool = [obj]
                else:
                    if( len(features) == len_of_file):
                        feature_pool = np.vstack([feature_pool, features])
                        label_pool.append(obj)
                n += 1
        mean = np.mean(feature_pool, axis=0)
        std = np.std(feature_pool, axis=0)
        feature_pool -= mean
        feature_pool /= std

        label_encoder = sklearn.preprocessing.LabelEncoder()
        label_encoder.fit(label_pool)
        encoded_labels = label_encoder.transform(label_pool)[:, np.newaxis]
        encoded_labels = np.squeeze(encoded_labels.T)

        classifier = sklearn.svm.SVC(kernel='linear', probability=True)
        # classifier = sklearn.tree.DecisionTreeClassifier()        
        # classifier = sklearn.ensemble.RandomForestClassifier(n_estimators=10)
        classifier.fit(feature_pool, encoded_labels)

        return ObjectClassifier(classifier, label_encoder, mean, std)

class Classfiy:

    def __init__(self,test_folder):
        self.test_folder = test_folder

    def load_classifier(self):
        return ObjectClassifier.load('home_clf.pkl','home_feature_clf.pkl')
        # self.clf = new.obj.load(home_clf,home_feature_clf)
        
    def load_test_file(self):
        files = np.array(glob.glob(self.test_folder + '/*'))
        n = 0
        for f in files:
            features = np.ravel(np.genfromtxt(f,delimiter=','))
            if n < 1 :
                len_of_file = len(features)
                feature_pool = np.array(features)
                m = re.search('../testFeatures/(.+?)_',f)
                if m:
                    true_labels = [m.group(1)]
            else:
                if(len(features) == len_of_file):
                    feature_pool = np.vstack([feature_pool, features])
                    m = re.search('../testFeatures/(.+?)_',f)
                    if m:
                        true_labels.append(m.group(1))
            n += 1

        return [feature_pool,true_labels]

    def classfiy_test_file(self,test_files):
        classifier = ObjectClassifier.load('home_clf.pkl','home_feature_clf.pkl')
        n = 0

        for f in test_files:
            # print f
            label,prob = classifier.classify(f)
            if n < 1 :
                all_labels = np.array(label)
            else:
                all_labels = np.vstack([all_labels, label])
            n += 1

        return all_labels

    def plot_confusion_matrix(self,cm,  labels, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')


if __name__ == "__main__":

    data_folder = "../features"
    test_folder = "../testFeatures"

    trainer = Trainer(data_folder)
    clf = trainer.train(objects='all')
    print "Training successful !!"

    print "Start testing"
    test = Classfiy(test_folder)
    [test_features , true_labels]= test.load_test_file()

    # for i in range(len(test_features)):
    #     print clf.classify(test_features[i])
    #     # print new_labels

    clf.save('home_clf.pkl','home_feature_clf.pkl')

    # new_clf = test.load_classifier()
    # print new_clf.classify(test_features[3])
    # print clf.classify(test_features[4])
    # [labels,probabilities] = 
    predicted_labels = test.classfiy_test_file(test_features)
    print "Testing complete, here are the results:"

    for i in range(len(predicted_labels)):
        print "Predicted Label " , i, " : ",predicted_labels[i], " True label : " , true_labels[i] 

    print "Accuracy is " , sklearn.metrics.accuracy_score(true_labels,predicted_labels)


    # Compute confusion matrix
    cm = sklearn.metrics.confusion_matrix(true_labels, predicted_labels)
    np.set_printoptions(precision=2)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    test.plot_confusion_matrix(cm,labels)
    plt.show()