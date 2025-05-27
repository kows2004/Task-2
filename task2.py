import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load Titanic dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 1. Generate summary statistics
print("Summary Statistics:\n", df.describe())

# 2. Create histograms and boxplots for numeric features
numeric_features = ["Age", "Fare", "Pclass"]

plt.figure(figsize=(12, 5))
for i, col in enumerate(numeric_features):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Histogram of {col}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
for i, col in enumerate(numeric_features):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df[col])
    plt.title(f"Boxplot of {col}")
plt.tight_layout()
plt.show()

# 3. Use pairplot and correlation matrix for feature relationships
sns.pairplot(df[numeric_features].dropna())
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df[numeric_features].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 4. Identify patterns, trends, anomalies
# Example: Survival rate by class
plt.figure(figsize=(6, 4))
sns.barplot(x="Pclass", y="Survived", data=df, ci=None)
plt.title("Survival Rate by Passenger Class")
plt.show()

# Interactive visualization using Plotly
fig = px.scatter(df, x="Age", y="Fare", color="Survived",
                 title="Age vs Fare (Colored by Survival)")
fig.show()

# 5. Make basic feature-level inferences
# Example: Checking gender-based survival rates
sns.barplot(x="Sex", y="Survived", data=df, ci=None)
plt.title("Survival Rate by Gender")
plt.show()
