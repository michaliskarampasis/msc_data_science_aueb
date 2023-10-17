#import libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

#function to create KDEplots for numeric features
def kdeplot(numerical_features,data_eda,colors):
    
    #create figure
    fig = plt.figure(figsize=(9,3), dpi=100)
    plt.title("KDE for {}".format(numerical_features))
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    
    #plot not-churned customers
    ax_kde = sns.kdeplot(data_eda[data_eda['churn'] == 'No'][numerical_features].dropna(), color= colors[0], label= 'Churn: No', shade='True')
    #plot churned customers
    ax_kde = sns.kdeplot(data_eda[data_eda['churn'] == 'Yes'][numerical_features].dropna(), color= colors[1], label= 'Churn: Yes', shade='True')
    ax_kde.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.4f}'))
    #ax_kde.set(xlim=(0, None))
    ax_kde.grid(False)
    plt.legend(['No Churn','Churn'])
	
    #save image
    plt.savefig('./images/2. EDA/KDE for {}.png'.format(numerical_features), bbox_inches='tight')
    
    return


#function to create countplots for numeric features
def countplot(numerical_features,data_eda,colors):
    
    #create figure
    fig = plt.figure(figsize=(15,5), dpi=100)
    plt.title("{} ~ Churn".format(numerical_features))
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    
    #create countplot for each numerical feature
    ax = sns.countplot(numerical_features,data = data_eda,hue = 'churn',palette = colors,edgecolor = 'black')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    plt.legend(['No Churn','Churn'])
    
    #save image
    plt.savefig('./images/2. EDA/{} ~ Churn.png'.format(numerical_features), bbox_inches='tight')
    
    return


#define function for plots
def barplot(categorical_features,data_eda,colors):
    
    #create figure
    fig = plt.figure(figsize=(15,5), dpi=100)
    plt.title("{} ~ Churn".format(categorical_features))
    fig.patch.set_facecolor('white')
    plt.style.use('classic')
    
    #create barplot for each numerical feature
    ax = sns.countplot(categorical_features,data = data_eda,hue = 'churn',palette = colors,edgecolor = 'black')
    #locate text
    for rect in ax.patches:
        ax.text(rect.get_x() + rect.get_width() / 2, 2*rect.get_height() /3, rect.get_height(), \
                horizontalalignment='center', fontsize = 11)
    plt.legend(['No Churn','Churn'])
    
    #save image
    plt.savefig('./images/2. EDA/{} ~ Churn.png'.format(categorical_features), bbox_inches='tight')
    
    return
