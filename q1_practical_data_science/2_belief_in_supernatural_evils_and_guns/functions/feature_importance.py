# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#define function for feature importance
def get_feature_importances(model,name,x_train,y_train):
    
    #train model
    model.fit(x_train, y_train)
    
    #feature names
    feature_names =  x_train.columns
    
    #create df with feature names and importances
    feature_importance = pd.DataFrame(feature_names, columns = ["feature"])
    
    #features importance %
    try:
        #if the model support feature_importances_
        importance = abs(model.feature_importances_)
        importance = (importance / importance.max()) * 100
    except AttributeError:
        #if the model does not support feature_importances_
        importance = abs(model.coef_[0])
        importance = (importance / importance.max()) * 100
    
    #add importances in df
    feature_importance["importance"] = importance
    
    # get the important features
    feature_importance = feature_importance.sort_values(by = ["importance"], ascending=False)
    #feature_importance = feature_importance.iloc[:10,:]
    
    # plot feature importance
    fig = plt.figure(figsize = (15,5),dpi=100)
    plt.style.use('classic')
    fig.patch.set_facecolor('white')
    
    sns.barplot(x="importance", y="feature", data=feature_importance,color = 'rosybrown')
    plt.title(f'Important Features ~ {name}')
    plt.xlabel('importance %')
    #save image
    fig.savefig(f'./graphs/Feature Importances/Important Features ~ {name}.png', bbox_inches='tight')
    
    return 
