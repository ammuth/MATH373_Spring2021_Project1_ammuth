#%% Import Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt

#%% Linear Regression
def linear_regression(Predictors, Response):
     XTX = Predictors.T @ Predictors
     XTy = Predictors.T @ Response
     
     beta = np.linalg.solve(XTX, XTy)
     
     return beta
   
#%% Binary Logistic Regression

#Sigmoid Binary Function
def sigmoid_binary(realnumber):
    Sigmoid = np.exp(realnumber)/(1 + np.exp(realnumber))
    return(Sigmoid)

#Cross-Entropy Binary Function
def crossentropy_binary(predictive, ground_truth):
    ce = ((-1*ground_truth) * np.log(predictive)) - ((1 - ground_truth) * np.log(1-predictive))
    return(ce)

#Loss Function
def L_fast(beta, X, y):
    y_pred = X @ beta
    
    N = X.shape[0]

    return (1/N)*np.sum((crossentropy_binary(sigmoid_binary(y_pred), y)))

#Gradient Loss Function
def grad_L(beta, X, y):
    N = X.shape[0]
    grad = 0
    
    for i in range(N):
        xiHat = X[i]
        yi = y[i]
                
        grad_i = (sigmoid_binary(np.vdot(xiHat, beta)) - yi)*xiHat
        
        grad += grad_i
    
    return grad/N

#Computing L to find lowest Objective Function Value per Iteration
def minimizeL(X, y):
    alpha = 1
    
    iterations = 50
    d_1 = X.shape[1]
    
    L_vals = np.zeros(iterations)
    beta_t = np.zeros((d_1))
        
    for t in range(iterations):
        L_vals[t] = L_fast(beta_t, X, y)
        
        print("Iteration: ", t, "Objective Function value: ", L_vals[t])
        
        beta_t = beta_t - alpha*grad_L(beta_t, X, y)
    
    return beta_t, L_vals


#%% Loading Datasets
titanic_train = pd.read_csv('C:\\Users\\lngsm\\Documents\\math373\\titanic_data\\train.csv', sep = ',', header = 0)
titanic_test = pd.read_csv('C:\\Users\\lngsm\\Documents\\math373\\titanic_data\\test.csv', sep = ',', header=0)

titanic_test1 = pd.read_csv('C:\\Users\\lngsm\\Documents\\math373\\titanic_data\\test.csv', sep = ',', header=0)
titanic_test1 = titanic_test1[['PassengerId']]

survived_train = titanic_train['Survived']

#%% Pre=processing data
def set_titles(titanic_data):
    #Take the Title of the person
    def substrings_in_string(big_string, substrings):
        for substring in substrings:
            if str.find(big_string, substring) != -1:
                return substring
        return np.nan
    
    title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                        'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',
                        'Don', 'Jonkheer']
    
    titanic_data['Title']=titanic_data['Name'].map(lambda x: substrings_in_string(x, title_list))
    
    def replace_titles(x):
        title=x['Title']
        if title in ['Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Countess', 'Mme']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms']:
            return 'Miss'
        elif title =='Dr':
            if x['Sex']=='Male':
                return 'Mr'
            else:
                return 'Mrs'
        else:
            return title
    titanic_data['Title'] = titanic_data.apply(replace_titles, axis=1)

def preprocess_data(titanic_data):     
    set_titles(titanic_data)
    
    titanic_data['Family_Size'] = (titanic_data['SibSp'] + titanic_data['Parch'])
    
    titanic_data = titanic_data[['Pclass', 'Sex', 'Age', 'Family_Size', 'Fare', 'Title', 'Embarked']]
    
    #Impute Values
    median_age = titanic_data['Age'].median()
    titanic_data['Age'].fillna(median_age, inplace=True)
    
    mean_fare = titanic_data['Fare'].mean()
    titanic_data['Fare'].fillna(mean_fare, inplace=True)    
    
    #One-hot encode values
    one_hot = pd.get_dummies(titanic_data['Pclass'])
    titanic_data = titanic_data.drop(['Pclass'], axis = 1)
    titanic_data = titanic_data.join(one_hot)
    
    one_hot = pd.get_dummies(titanic_data['Title'])
    titanic_data = titanic_data.drop(['Title'], axis = 1)
    titanic_data = titanic_data.join(one_hot)
    
    one_hot = pd.get_dummies(titanic_data['Embarked'])
    titanic_data = titanic_data.drop(['Embarked'], axis = 1)
    titanic_data = titanic_data.join(one_hot)
    
    titanic_data = titanic_data.drop('Sex', axis=1)
    
    return titanic_data

#%% Diabetes Dataset - Linear Regression
diabetes = load_diabetes()

x_diabetes = diabetes.data

y_diabetes = diabetes.target

# Create X dataframe with diabetes data + Ones for Beta[0] coefficient
nrow = x_diabetes.shape[0]

X = np.zeros((nrow, 11))
    
X[ : , 0] = np.ones(nrow)
X[ : , 1] = x_diabetes[ : , 0]
X[ : , 2] = x_diabetes[ : , 1]
X[ : , 3] = x_diabetes[ : , 2]
X[ : , 4] = x_diabetes[ : , 3]
X[ : , 5] = x_diabetes[ : , 4]
X[ : , 6] = x_diabetes[ : , 5]
X[ : , 7] = x_diabetes[ : , 6]
X[ : , 8] = x_diabetes[ : , 7]
X[ : , 9] = x_diabetes[ : , 8]
X[ : , 10] = x_diabetes[ : , 9]

beta_linearregression = linear_regression(X, y_diabetes)

#Predict Values based on data
yivalspred = (beta_linearregression[0]*X[ : , 0] + beta_linearregression[1]*X[ : , 1] + 
    beta_linearregression[2]*X[ : , 2] + beta_linearregression[3]*X[ : , 3] +
    beta_linearregression[4]*X[ : , 4] + beta_linearregression[5]*X[ : , 5] +
    beta_linearregression[6]*X[ : , 6] + beta_linearregression[7]*X[ : , 7] +
    beta_linearregression[8]*X[ : , 8] + beta_linearregression[9]*X[ : , 9] + 
    beta_linearregression[10]*X[ : , 10])


#Plot Predicted vs Actual
plt.figure()
plt.scatter(yivalspred, y_diabetes)

print("Beta values of Linear Regression: " + str(beta_linearregression))

#%% Titanic Dataset - Binary Logistic Regression
titanic_train = preprocess_data(titanic_train)

titanic_train = titanic_train.values

#Standardize the data
scaler = StandardScaler()
titanic_train = scaler.fit_transform(titanic_train) 

titanic_train = np.insert(titanic_train, 0, 1, axis=1)

beta_est, L_vals = minimizeL(titanic_train, survived_train)

plt.plot(L_vals)

print("Beta Estimate: ", beta_est)

### Predictions
titanic_test = preprocess_data(titanic_test)
titanic_test = titanic_test.values
scaler = StandardScaler()
titanic_test = scaler.fit_transform(titanic_test) 

titanic_test = np.insert(titanic_test, 0, 1, axis=1)

predictions = np.round(sigmoid_binary(titanic_test@beta_est))

titanic_test1['Survived']=pd.Series(predictions)
titanic_test1['Survived'] = titanic_test1['Survived'].astype(int)

titanic_test1.to_csv('Titanic Submission.csv', sep=',', index=False)
