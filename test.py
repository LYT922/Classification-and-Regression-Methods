import pandas as pd
import numpy as np

df = pd.read_csv('penguins.csv')
df=df.dropna()
del df["year"]
print(df)

main_statistics=df.describe()
print(main_statistics)

df['species'] = pd.Categorical(df['species'])
df['island'] = pd.Categorical(df['island'])
df['sex'] = pd.Categorical(df['sex'])
df['species']=df['species'].cat.codes
df['island']=df['island'].cat.codes
df['sex']=df['sex'].cat.codes


df_normal=df
features = df_normal[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]
min_value=features.min()
max_value=features.max()
features=(features-min_value)/(max_value-min_value)
df_normal[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']] = features
print(df_normal)

bias=df_normal.shape[0]*[1]
df_normal.insert(loc=0, column='bias', value=bias)

X=df_normal.drop(['island'], axis=1)
y=df_normal[['island']] #target value

df = df.sample(frac = 1) #shuffle data
train_size = int(0.8 * len(df_normal))
train_X = X[:train_size]
train_y= y[:train_size]
test_X=X[train_size:]
test_y=y[train_size:]

t_X_shpe=train_X.shape
t_y_shpe=train_y.shape
tes_X_shpe=test_X.shape
tes_y_shpe=test_y.shape
print(t_X_shpe,t_y_shpe,tes_X_shpe,tes_y_shpe)

class LogitRegression():
    def __init__(self, learning_rate=1e-6, num_iterations=10000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))
    
    def cost(self, X, y):
        n = y.shape[0]
        h = self.sigmoid(np.dot(X, self.w.T) )
        J = (1 / n) * (-y.iloc[:,0]*np.log(h)-(1-y.iloc[:,0])*np.log(1-h))
        return J
    
    def gradient_descent(self, X, y):
        n = y.shape[0]
        pred = self.sigmoid(np.dot(X, self.w.T))
        delta = pred - y.iloc[:,0]
        dw = (1 / n) * np.dot(X.T, delta)
        return dw

    def fit(self, X, y):
        # Initialize the weights
        self.w = np.random.uniform(0, 1, X.shape[1])
        # Initialize the loss array
        self.loss = []
        # Run gradient descent for the specified number of iterations
        for i in range(self.num_iterations):
            dw = self.gradient_descent(X, y)
            self.w -= self.learning_rate * dw
            # Calculate the cost function and append it to the loss array
            J = self.cost(X, y)
            self.loss.append(J)

    def predict(self, X):
        # Predict the binary labels for the input data
        y_pred = self.sigmoid(np.dot(X, self.w))
        return np.where(y_pred >= 0.5, 1, 0)
    

model = LogitRegression()

model.fit(train_X, train_y)

print(model.predict(test_X))
print(test_y)