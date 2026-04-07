from flask import Flask, render_template, request
import pickle
from flask_mail import Mail, Message
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')
app.config['MAIL_USE_TLS'] = True

mail = Mail(app)

with open('model.pkl', 'rb') as file:
    clf = pickle.load(file)

@app.route('/', methods=['GET', 'POST'])
def home():
    #'WaterLogging', 'CleanWater', 'HouseCleaning', 'Macchardani', 'InfectionProb'
    if request.method == 'POST':
        water_logging = int(request.form['water_logging'])
        clean_water = int(request.form['clean_water'])
        house_cleaning = int(request.form['house_cleaning'])
        macchardani = int(request.form['macchardani'])



        inputFeature = [[water_logging, clean_water, house_cleaning, macchardani]]
        infPro = clf.predict_proba(inputFeature)[0][1]

        return render_template('show.html', inf=round(infPro * 100))
    return render_template('index.html')

@app.route('/about', methods=["GET", "POST"])
def About():
    return render_template('about.html')

#@app.route('/contact', methods=["GET", "POST"])
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        msg_body = request.form.get('message')

        msg = Message(subject=f"New Contact from {name}",
                      sender=email,
                      recipients=['manishkumar96963172@gmail.com'],
                      body=f"From: {name} <{email}>\n\n{msg_body}")
        mail.send(msg)
        return "Message Sent!"
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)



