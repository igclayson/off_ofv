import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

csv_file='toy_model_example.csv'
#csv_file='Cu_6MR_PBE_3NN.csv'
df = pd.read_csv(csv_file, header =0)
X=df.drop(columns=['step','Relative energy / kJ mol^-1'])
Y=df.iloc[:, -1] # target variable is on the right-most side
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.33, random_state=42) # 2/3 train and 1/3 test
randomForest = RandomForestRegressor(n_estimators = 5000, random_state = 42) # untuned hyperparameters
randomForest.fit(X_train, Y_train)
Y_hat = randomForest.predict(X_test)
print("Actual test data values\t\tPredicted test data values\t\tAbsolute error\t\t\tSquared error") # extra tabs for nice spacing
for j in range(len(Y_hat)):
	Y_test_j=Y_test.iloc[j]; Y_hat_j=Y_hat[j]
	print("{:.5f}\t\t\t\t{:.5f}\t\t\t{:.5f}\t\t\t{:.5f}".format(Y_test_j,Y_hat_j,abs(Y_hat_j - Y_test_j),(Y_hat_j - Y_test_j)**2))
