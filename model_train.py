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


fixed_length = 84 


data = np.array([seq[:fixed_length] if len(seq) > fixed_length else np.pad(seq, (0, fixed_length - len(seq))) for seq in raw_data])


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, shuffle=True, stratify=labels)


model = RandomForestClassifier()
model.fit(x_train, y_train)


y_predict = model.predict(x_test)


score = accuracy_score(y_test, y_predict)
print('{}% of samples were classified correctly!'.format(score * 100))


# cm = confusion_matrix(y_test, y_predict)

# # Plot the confusion matrix
# plt.figure(figsize=(10, 7))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(labels), yticklabels=np.unique(labels))
# plt.xlabel('Predicted Labels')
# plt.ylabel('True Labels')
# plt.title('Confusion Matrix')
# plt.show()


print(classification_report(y_test, y_predict, target_names=[str(label) for label in np.unique(labels)]))


with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
print("Complete")