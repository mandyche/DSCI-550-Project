import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


st.title("Exploring the Impact of Remote Work on Employee Performance and Satisfaction")


### Data Cleaning and Preprocessing

# Load the dataset
df = pd.read_csv('Extended_Employee_Performance_and_Productivity_Data.csv')

# Display basic info
st.write("### Dataset Overview")
st.write(df.info())
st.write(df.describe())

# Handle missing values in numeric columns only
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

### Exploratory Data Analysis (EDA)

# Define performance categories
performance_bins = [0, 50, 75, 100]
performance_labels = ['Low', 'Mid', 'High']
df['Performance_Category'] = pd.cut(df['Performance_Score'], bins=performance_bins, labels=performance_labels)

#(Visualize Remote Work Distribution Across Performance Categories)
st.write("### Remote Work Frequency Across Performance Categories")
fig, ax = plt.subplots()
sns.countplot(x='Performance_Category', hue='Remote_Work_Frequency', data=df, ax=ax)
plt.title('Remote Work Frequency Across Performance Categories')
plt.xlabel('Performance Category')
plt.ylabel('Count')
st.pyplot(fig)

#(Correlation Analysis of Remote Work, Satisfaction, and Performance)
st.write("### Correlation Analysis")
correlation_matrix = df[['Remote_Work_Frequency', 'Employee_Satisfaction_Score', 'Performance_Score']].corr()

fig, ax = plt.subplots()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
plt.title('Correlation Between Remote Work, Satisfaction, and Performance')
st.pyplot(fig)


### Predictive Modeling


# Define features and target
X = df[['Remote_Work_Frequency', 'Employee_Satisfaction_Score', 'Performance_Score', 'Overtime_Hours']]
y = df['Resigned']  # Assume this column indicates resignation status

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

st.write("### Model Evaluation")
st.write(f"Accuracy: {accuracy:.2f}")
st.text("Classification Report:\n" + report)