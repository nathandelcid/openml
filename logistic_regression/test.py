from log_reg import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from min_max_scaler import MinMaxScaler

log_reg = LogisticRegression()
scaler = MinMaxScaler()

data = load_iris()
X = data.data
y = data.target

scaler = MinMaxScaler()
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = scaler.fit_transform(X_train)

# Only use the first 10 samples for demonstration purposes
X_train = X_train[:10]
y_train = y_train[:10]


log_reg.fit(X_train, y_train)