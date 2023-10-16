# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from statsmodels.formula.api import logit
import statsmodels.formula.api as smf

# function to run logistic regression
def run_logistic_regression(df, col, group,i):
    
    """
    Run logistic regression for a given dependent variable and group of columns.
    Returns a dataframe with coefficient results.
    """
    
    # create a df with relevant columns 
    df_reg = df[[col] + group]
    
    # create the formula for logistic regression
    all_variables_formula = f"{col} ~ " + '+'.join(group)
    
    # fit the logistic regression model
    logit_mod = logit(formula=all_variables_formula, data=df_reg)
    logit_res = logit_mod.fit(disp=False)

    # extract coefficients and p-values
    coef_pvals = logit_res.summary2().tables[1][['Coef.', 'P>|z|']]
    
    # apply significance stars to coefficients
    # *p < 0.05, **p < 0.01, ***p < 0.001
    result = coef_pvals.apply(lambda x: f"{np.exp(x['Coef.']):.2f}***" if x['P>|z|'] < 0.001 else 
                                       (f"{np.exp(x['Coef.']):.2f}**" if x['P>|z|'] < 0.01 else
                                       (f"{np.exp(x['Coef.']):.2f}*" if x['P>|z|'] < 0.05 else f"{np.exp(x['Coef.']):.2f}")
    ), axis=1)
    
    # add results in a dataframe
    df_result = pd.DataFrame(result, columns=[f"{col}_{i+1}"])
    
    return df_result


# function to run linear regression
def run_linear_regression(df, col, group,i):
    
    """
    Run linear regression for a given dependent variable and group of columns.
    Returns a dataframe with coefficient results.
    """
    
    # create a df with relevant columns 
    df_reg = df[[col] + group]
    
    # create the formula for linear regression
    all_variables_formula = f"{col} ~ " + '+'.join(group)
    
    # fit the linear regression model
    linear_mod = smf.ols(formula=all_variables_formula, data=df_reg)
    linear_mod = linear_mod.fit(disp=False)

    # extract coefficients and p-values
    coef_pvals = linear_mod.summary2().tables[1][['Coef.', 'P>|t|']]
    
    # apply significance stars to coefficients
    # *p < 0.05, **p < 0.01, ***p < 0.001
    result = coef_pvals.apply(lambda x: f"{np.exp(x['Coef.']):.2f}***" if x['P>|t|'] < 0.001 else 
                                       (f"{np.exp(x['Coef.']):.2f}**" if x['P>|t|'] < 0.01 else
                                       (f"{np.exp(x['Coef.']):.2f}*" if x['P>|t|'] < 0.05 else f"{np.exp(x['Coef.']):.2f}")
    ), axis=1)
    
    # add results in a dataframe
    df_result = pd.DataFrame(result, columns=[f"{col}_{i+1}"])
    
    return df_result
