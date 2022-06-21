import pickle
import numpy as np 
from flask import Flask, render_template, request, redirect, url_for


loaded_model = pickle.load(open('finalized_model.pkl', 'rb'))
print('model loaded')

X_cols = pickle.load(open('col.pkl', 'rb'))

print("cols loaded")

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X_cols==location)[0][0]
    x = np.zeros(len(X_cols))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    return loaded_model.predict([x])[0]



app = Flask(__name__,template_folder='templates')



@app.route("/")
def hello_world():
    return render_template('first.html')

@app.route("/predict",methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        bhk = request.form.get('bhk')
        location = request.form.get('location')
        area = request.form.get('flat_area')
        bath = request.form.get('bath')
        # print(location, bhk, bath, area)
        try:
            res = predict_price(location, area, bath, bhk)
            return render_template("first.html", prediction_text = f"The predicted price is: {round(res, 3)} Lakhs.")
        except:
            return render_template("first.html", prediction_text = f"Sorry location is not available.")
    else:
        return redirect(url_for(''))
        



if __name__ == '__main__':
    app.run(debug=True)

    # localhost:5000/predict
