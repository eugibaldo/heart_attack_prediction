import matplotlib.pyplot as plt
import os
import seaborn as sns
import scipy.stats as stats
import numpy as np
import pandas as pd
plt.rcParams['font.size']=8
plt.rcParams['figure.figsize']=(16,9)



def visualize_hist(data,columns,density=True):

    '''
    this function display a histogram for each feature
    :param data: dataframe
    :param columns: features to use in histograms
    :param density: if I want the histograms with density(true) or frequency(false)
    :return:
    '''
    
    fig,axes=plt.subplots(nrows=2,ncols=7)
    colors=['green','blue','yellow','orange','brown','#DDDABB','#CCDDAA','#BDAABB','#BACBAF','#FDBDEA','#BEECBA','#ADCBDF','#EEEEEE','#FFCCBC']
    fig.suptitle('histograms for all the features and the target',fontsize=12,fontweight='bold')


    for i,col in enumerate(columns):
        
        if i<7:
            
            axes[0,i].hist(data[col],density=density,color=colors[i])
            axes[0,i].set_title(col,fontweight='bold',fontsize=10)
            axes[0, i].tick_params(labelleft=False,left=False)

            if i==0:

                axes[0,0].set_ylabel('density')


            
        else:
            
            axes[1,i-7].hist(data[col],density=density,color=colors[i])
            axes[1,i-7].set_title(col,fontweight='bold',fontsize=10)
            axes[1, i-7].tick_params(labelleft=False,left=False)

            if i==7:

                axes[1,0].set_ylabel('density')



    plt.savefig(os.path.join(os.getcwd(),'histograms_features.png'))
    plt.show()
    
    return

def heatmap(data,columns,categorical=False):
    '''
    this function displays the heatmap for continous and nominal data
    :param data: dataframe
    :param columns: numerical columns(continous data) or nominal columns(discrete data)
    :param categorical: if True I will perform association using Cramer V, otherwise pearson correlation
    :return:
    '''

    try:

       sub_data=data.loc[:,columns]
       #here I compute the pearson correlation coefficient among the continous variables
       if categorical==False:

           plt.figure(figsize=(12,9))
           plt.title('pearson correlation among numerical variables',fontsize=12)
           sns.heatmap(sub_data.corr(),annot=True)
           plt.savefig(os.path.join(os.getcwd(), 'correlations.png'))
           plt.show()
       #here I compute the association between nominal variables
       else:
           association_dict={} #store in a dictionary the association between pairwise features using cramer v
           for i in range(len(columns)):
               x=sub_data.iloc[:,i]
               association_dict[columns[i]]=[]
               for j in range(len(columns)):
                   y=sub_data.iloc[:,j]
                   confusion_matrix = pd.crosstab(x, y)
                   chi2 = stats.chi2_contingency(confusion_matrix)[0]
                   n = confusion_matrix.sum().sum()
                   phi2 = chi2 / n
                   r, k = confusion_matrix.shape
                   phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                   rcorr = r - ((r - 1) ** 2) / (n - 1)
                   kcorr = k - ((k - 1) ** 2) / (n - 1)
                   res=np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
                   association_dict[columns[i]].append(res)

           association_df=pd.DataFrame.from_dict(association_dict,orient='index')
           association_df.columns=association_df.index
           plt.figure(figsize=(12, 9))
           plt.title('association among nominal variables', fontsize=12)
           sns.heatmap(association_df, annot=True,cmap='crest')
           plt.savefig(os.path.join(os.getcwd(), 'associations.png'))
           plt.show()



    except KeyError as e:

        print(e)


    return

def feature_importance_hist(pipeline):

    '''
    this function displays the weights of the logistic regression model
    :param pipeline: pipeline with PCA and logistic regression model
    :return: ordered coefficient of logistic model
    '''


    coefs = pipeline[-1].coef_[0]
    plt.style.use('ggplot')
    plt.figure(figsize=(16,9))
    plt.title('feature importance for logistic regression model')
    plt.xlabel('feature')
    plt.ylabel('weight contribution')
    plt.bar([i for i in range(len(coefs))],coefs)
    plt.savefig(os.path.join(os.getcwd(), 'coefficients.png'))
    plt.show()

    return coefs

