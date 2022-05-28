import pandas as pd
import numpy as np

data = pd.read_excel("C:/Data/red.xlsx")
data.columns


data= data.drop(["patient_ID"], axis=1)
data= data.drop(["patient_age"], axis=1)
data= data.drop(["gender"], axis=1)
data= data.drop(["test_booking_date"], axis=1)
data= data.drop(["Sample_Collection_Date"], axis=1)
data= data.drop(["Mode_Of_Transport"], axis=1)
data.columns

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data2=data.copy()
data2["test_name"] = le.fit_transform(data["test_name"])
data2["sample"] = le.fit_transform(data2["sample"])
data2["sample_storage"] = le.fit_transform(data2["sample_storage"])
data2["Cut_off_Schedule"] = le.fit_transform(data2["Cut_off_Schedule"])
data2["traffic_conditions"] = le.fit_transform(data2["traffic_conditions"])
data2["Reached_On_Time"] = le.fit_transform(data2["Reached_On_Time"])

# Input and Output Split
predictors = data2.loc[:, data2.columns!='Reached_On_Time']
type(predictors)


target = data2['Reached_On_Time']
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=2)


########################## Adaboost ##########################################

from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.05, n_estimators = 500)

ada_clf.fit(x_train, y_train)

from sklearn.metrics import accuracy_score, confusion_matrix


# Evaluation on Testing Data
confusion_matrix(y_test, ada_clf.predict(x_test))
accuracy_score(y_test, ada_clf.predict(x_test))

# Evaluation on Training Data
accuracy_score(y_train, ada_clf.predict(x_train))

final_df = pd.concat([predictors,target], axis=1)

input_data = (6,0,1,16.1,10.15,0,13.15,4,0,12,72,3,9,54)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = ada_clf.predict(input_data_reshaped)
print(prediction)

if prediction == 1:
    print("Reached on time")
else:
    print("Dont reach on time")



import pickle
filename = "train.pkl"
pickle.dump(ada_clf, open(filename,"wb"))

filename1 = "final.pkl"
pickle.dump(data, open(filename1,"wb"))
