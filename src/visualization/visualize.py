import sklearn.metrics as metr
import matplotlib.pyplot as plt
def classification_reports(classifier,X,y, verbose = False):
    """
    Retrieved from: https://www.kaggle.com/asiyehbahaloo/asiyeh-bahaloo
    
    Provides Confusion matrix, accuracy, AUC and standard classification report. Note Verbose returns ROC curve values and accuracy
    
    """
    y_pred = classifier.predict_proba(X)[:,1]
    y_pred_lab = classifier.predict(X)
    size_data = len(y)
    count_class_1 = sum(y)
    count_class_0 = size_data - count_class_1
    print(' class 1 : ', count_class_1)
    print(' class 0 : ', count_class_0)
    fpr, tpr, thresholds = metr.roc_curve(y, y_pred)
    print("Confusion Matrix: \n",metr.confusion_matrix(y,y_pred_lab))
    score=metr.accuracy_score(y,y_pred_lab)
    print("Accuracy: ",score)
    auc=metr.roc_auc_score(y,y_pred)
    print("AUC: ",auc)
    print(metr.classification_report(y,y_pred_lab))
    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot( fpr, tpr,color='darkorange')
    plt.show()
    if verbose:
        return fpr,tpr,score