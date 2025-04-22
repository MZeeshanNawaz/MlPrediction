from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os

app = Flask(__name__)
data = {
    'Day':['sunny','windy','sunny','sunny','windy','sunny','sunny','windy','sunny','windy'],
    'Temp':['hot','cool','hot','cool','hot','cool','hot','cool','hot','cool'],
    'class':['play','Not play','play','Not play','play','play','play','Not play','play','Not play']
}
df = pd.DataFrame(data)
X_raw = df[['Day','Temp']]
Y_raw = df['class']
Onehot_encoder = OneHotEncoder()
x_encoded = Onehot_encoder.fit_transform(X_raw).toarray()

label_Encoder = LabelEncoder()
y_encoded = label_Encoder.fit_transform(Y_raw)

x_train,x_test, y_train ,y_test = train_test_split(x_encoded,y_encoded,test_size=0.3,random_state=42)
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Save modal and encoder
joblib.dump(model,'model.pkl')
joblib.dump(Onehot_encoder,'onehot.pkl')
joblib.dump(label_Encoder,'label.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        day = request.form['day']
        temperature = request.form['temperature']

        model = joblib.load('model.pkl')
        onehot = joblib.load('onehot.pkl')
        label = joblib.load('label.pkl')

        new_data = pd.DataFrame([[day,temperature]],columns=['Day','Temp'])
        new_encoded = onehot.transform(new_data).toarray()

        prediction = model.predict(new_encoded)
        result = label.inverse_transform(prediction)[0]

        return render_template('index.html',prediction_text = f'Prediction: {result}')
if __name__ == '__main__':
    app.run(debug=True)
