import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("mail_data.csv", encoding="ISO-8859-1")

# Drop duplicate rows if any
df = df.drop_duplicates()

# Map categories to numeric labels: spam = 0, ham = 1
df.loc[df["Category"] == "spam", "Category"] = 0
df.loc[df["Category"] == "ham", "Category"] = 1

# Prepare feature and target variables
X = df['Message'].astype(str)
y = df['Category'].astype(int)

# Split data into train and test (optional but good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the TfidfVectorizer on training data
vectorizer = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = vectorizer.fit_transform(X_train)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, y_train)

# Save the fitted vectorizer to a file
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# Save the trained model to a file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ vectorizer.pkl and model.pkl have been created successfully!")
