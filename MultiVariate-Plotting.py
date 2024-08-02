import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the data with additional options
file_path = 'D:/AI&ML/titanic.csv'
data = pd.read_csv(file_path, na_values=['NA', '?'], dtype={'Pclass': 'category', 'Survived': 'category', 'Sex': 'category', 'Embarked': 'category'})

# Preprocessing the data
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Convert categorical variables to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Handling missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Detect and handle outliers in the 'Fare' column
q1 = data['Fare'].quantile(0.25)
q3 = data['Fare'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
data['Fare'] = np.where(data['Fare'] > upper_bound, upper_bound, np.where(data['Fare'] < lower_bound, lower_bound, data['Fare']))

# Standardize the numeric features
scaler = StandardScaler()
data[['Age', 'Fare']] = scaler.fit_transform(data[['Age', 'Fare']])

# Ensure all data types are numeric
data = data.apply(pd.to_numeric)

# Create pairplot for multivariate distribution
pairplot = sns.pairplot(data, hue='Survived', plot_kws={'alpha':0.5})

# Add titles and labels
for ax in pairplot.axes.flatten():
    ax.set_xlabel(ax.get_xlabel(), fontsize=10)
    ax.set_ylabel(ax.get_ylabel(), fontsize=10)
    ax.set_title(f'{ax.get_ylabel()} vs {ax.get_xlabel()}', fontsize=12)
plt.tight_layout()
plt.show()
