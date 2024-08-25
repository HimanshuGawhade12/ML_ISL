import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data_dict = pickle.load(open('./data.pickle', 'rb'))

raw_data = data_dict['data']
labels = np.asarray(data_dict['labels'])

# Define a fixed length for truncation
fixed_length = 84  # This should match the length used during model training

# Truncate or pad sequences to the fixed length
data = np.array([seq[:fixed_length] if len(seq) > fixed_length else np.pad(seq, (0, fixed_length - len(seq))) for seq in raw_data])

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)

# Initialize and train the RandomForestClassifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict the labels for the test set
y_predict = model.predict(x_test)

# Evaluate the model's accuracy
score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))

# # Compute the confusion matrix
# cm = confusion_matrix(y_test, y_predict)

# # Plot the confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()

# Print classification report
print(classification_report(y_test, y_predict, target_names=[str(label) for label in np.unique(labels)]))

# Save the trained model to a file
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Complete")