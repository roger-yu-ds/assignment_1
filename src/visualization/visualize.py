import sklearn.metrics as metr

def classification_reports(classifier,X,y):
    """
    Retrieved from: https://www.kaggle.com/asiyehbahaloo/asiyeh-bahaloo
    
    Provides Confusion matrix, accuracy, AUC and standard classification report
    
    """
    
    size_data = len(labels)
    count_class_1 = sum(labels)
    count_class_0 = size_data - count_class_1
    print(' class 1 : ', count_class_1)
    print(' class 0 : ', count_class_0)
    y = classifier.predict(features)
    fpr, tpr, thresholds = metr.roc_curve(labels, y)
    print("Confusion Matrix: \n",metr.confusion_matrix(labels,y))
    score=metr.accuracy_score(labels,y)
    print("Accuracy: ",score)
    auc=metr.roc_auc_score(labels,y)
    print("AUC: ",auc)
    print(metr.classification_report(labels,y))
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot( fpr, tpr,color='darkorange')
    plt.show()

    return fpr,tpr,score