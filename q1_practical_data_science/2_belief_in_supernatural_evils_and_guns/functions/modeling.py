# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.model_selection import  StratifiedKFold, cross_validate, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix
from sklearn import metrics



# define function for cross validation
def cross_validation_of_models(models,x_train,y_train):    
    
    #create a dataframe to store the results
    index=[model[0] for model in models]
    columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC']
    cv_scores = pd.DataFrame(np.nan, index=index, columns=columns)
    
    #loop through classifiers
    for i,(name, model) in enumerate(models):
        
        #define kfolds 
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        scoring = ['accuracy','precision','recall','f1','roc_auc']
            
        #calculate scores for each model
        scores = cross_validate(model, x_train, y_train, 
                        cv = cv, scoring=scoring, error_score="raise")
            
        #evaluate
        avg_accuracy = scores['test_accuracy'].mean()
        avg_precision = scores['test_precision'].mean()
        avg_recall = scores['test_recall'].mean()
        avg_f1 = scores['test_f1'].mean()
        avg_roc = scores['test_roc_auc'].mean()
        
        
        #store results
        cv_scores.loc[name, 'Accuracy'] = avg_accuracy
        cv_scores.loc[name, 'Precision'] = avg_precision
        cv_scores.loc[name, 'Recall'] = avg_recall
        cv_scores.loc[name, 'F1 Score'] = avg_f1
        cv_scores.loc[name, 'ROC'] = avg_roc
    
    #sort by F1 Score
    cv_scores = cv_scores.sort_values(by='Accuracy', ascending=False)
    
    return cv_scores


#define function for hyperparameter tuning
def hyperparameter_tuning(models,param_grid,x_train, y_train):
    
    # create a dataframe to store the results
    rs_scores = pd.DataFrame(np.nan, index=[model[0] for model in models], columns=['Accuracy'])

    # loop through classifiers
    for index, model in enumerate(models):
        
        # RandomizedSearchCV
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        rs = GridSearchCV(model[1], param_grid[index][1], scoring='accuracy', cv=cv, n_jobs=-1)
        rs.fit(x_train, y_train)
        
        # evaluate
        best_score = rs.best_score_
        best_params = rs.best_params_
        best_estimator = rs.best_estimator_
        
        # print results
        print('='*106)
        print(f'{model[0]}')
        print('-'*106)
        print(f'Best grid score: {best_score}')
        print(f'Best grid params: {best_params}')
        print(f'Best estimator: {best_estimator}', end='\n\n')
        
        # store the results
        rs_scores.loc[model[0], 'Accuracy'] = best_score
    
    return rs_scores


#define function for evaluate the results
def model_evaluation(y_test, y_pred,name, model,prediction_results,i):
    
    #calculate metrics for evaluation
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # print results
    print('='*125)
    print(f'# {i+1} - {name}')
    print('='*125)
    
    # store results
    prediction_results.loc[name, 'Precision'] = precision
    prediction_results.loc[name, 'Recall'] = recall
    prediction_results.loc[name, 'F1 Score'] = f1
    prediction_results.loc[name, 'ROC'] = roc
    prediction_results.loc[name, 'Accuracy'] = accuracy
    
    print(pd.DataFrame(prediction_results.loc[f'{name}',:]).T)
        
    return prediction_results


#define function to evaluate the performance between classifiers
def classification_prediction(models, x_train, x_test, y_train, y_test):
    
    #create a dataframe to store the results
    index=[model[0] for model in models]
    columns=['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC']
    prediction_results = pd.DataFrame(np.nan, index=index, columns=columns)
    
    #loop through classifiers
    for index, (name,model) in enumerate(models):
        
        #fit
        clf = model.fit(x_train, y_train)
        
        #predict
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[::,1]
                
        #classification report
        prediction_results = model_evaluation(y_test, y_pred, name, model,prediction_results,index)
        
        #create figure
        fig = plt.figure(figsize=(12,4),dpi=100,facecolor='white')
        
        #compute and plot the confusion matrix
        cm = confusion_matrix(y_test,y_pred)
        names = ['True Negative','False Positive','False Negative','True Positive']
        counts = [value for value in cm.flatten()]
        percentages = ['{0:.2%}'.format(value) for value in cm.flatten()/np.sum(cm)]
        labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(names,counts,percentages)]
        labels = np.asarray(labels).reshape(2,2)
        
        ax1 = fig.add_subplot(121)
        sns.heatmap(cm,annot = labels,cmap ='rocket',fmt ='', ax=ax1)

        #create ROC curve
        ax2 = fig.add_subplot(122)
        #define metrics
        #y_pred_proba = clf.predict_proba(x_test)[::,1]
        fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
        
        #plot
        ax2.plot(fpr,tpr,color='darkorange',lw=1)
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        ax2.set_title('ROC Curve')
        fig.suptitle(f'{name}')
        plt.show()
        #save image
        fig.savefig(f'./graphs/Modeling/{name}.png', bbox_inches='tight')
        
    #sort by Accuracy Score
    prediction_results = prediction_results.sort_values(by=['Accuracy','F1 Score','Recall'], ascending=False)
    
    return prediction_results