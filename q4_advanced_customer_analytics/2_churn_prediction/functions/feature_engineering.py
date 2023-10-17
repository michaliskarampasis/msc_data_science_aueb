#import libraries
import matplotlib.pyplot as plt
import seaborn as sns


#function for distplot
def distplot(feature, frame, color='rosybrown',when='before scaling'):
    
    #create figure
    fig = plt.figure(figsize = (7,3),dpi=100)
    plt.style.use('classic')
    fig.patch.set_facecolor('white')
    plt.title("Distribution for {}".format(feature))
    
    #distplot
    sns.distplot(frame[feature], color= color)
    
    #save image
    fig.savefig(f'./images/4. Feature Engineering/Distribution for {feature} {when}.png', bbox_inches='tight')
    
    return
