from flask import Flask, render_template, request, redirect, url_for, jsonify
from keras.preprocessing import image as keras_image
from keras.models import load_model
import matplotlib.pyplot as plt
from datetime import datetime
from io import BytesIO
import yfinance as yf
from PIL import Image, ImageOps
import pandas as pd
import numpy as np
import requests
import sqlite3
import base64
import os
import re
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical


app = Flask(__name__)

def model_init():
    def prepare_data():
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # Convert to grayscale by reshaping and normalizing
        x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

        num_classes = 10
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        return x_train, y_train, x_test, y_test

    # Get prepared data
    x_train, y_train, x_test, y_test = prepare_data()

    # Build and compile the model
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')  # Assuming 10 classes for MNIST
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model with the prepared data
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

    # Save the model
    model.save('model.h5')

def load_model_or_train():
    try:
        global model
        model = load_model('model.h5')
    except:
        model_init()

load_model_or_train()



@app.route('/')
def accueil():
    return render_template('Accueil.html')

@app.route('/formulaire_view')
def formulaire_view():
    return render_template('Form.html')

def get_db_connection():
    conn = sqlite3.connect('data_base.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    check_table_query = '''
        SELECT name FROM sqlite_master WHERE type='table' AND name='User'
    '''
    cursor.execute(check_table_query)
    table_exists = cursor.fetchone()

    if not table_exists:
        create_table_query = '''
            CREATE TABLE User (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prenom TEXT NOT NULL,
                nom TEXT NOT NULL,
                sexe TEXT NOT NULL,
                username TEXT NOT NULL UNIQUE
            )
        '''
        
        cursor.execute(create_table_query)
        conn.commit()
    check_table_query = '''
        SELECT name FROM sqlite_master WHERE type='table' AND name='Logs'
    '''
    cursor.execute(check_table_query)
    table_exists = cursor.fetchone()

    if not table_exists:
        create_table_query = '''
            CREATE TABLE Logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                datetime DATE NOT NULL,
                ticker TEXT NOT NULL,
                company TEXT
            )
        '''
        
        cursor.execute(create_table_query)
        conn.commit()
    conn.close()

    return conn
get_db_connection()


@app.route('/formulaire_completed', methods=['POST'])
def formulaire_client():
    if request.method == 'POST':
        prenom = request.form['prenom']
        nom = request.form['nom']
        sexe = request.form['sexe']
        username = request.form['username']

        # Connexion à la base de données
        conn = sqlite3.connect('data_base.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Vérification si le username existe déjà
        cursor.execute("SELECT * FROM User WHERE username = ?", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            return "Ce username est déjà utilisé. Veuillez en choisir un autre."

        # Insertion des données dans la base de données
        cursor.execute("INSERT INTO User (prenom, nom, sexe, username) VALUES (?, ?, ?, ?)",
                       (prenom, nom, sexe, username))
        conn.commit()
        conn.close()

        return redirect(url_for('get_clients'))
    
    return render_template("Form.html")

@app.route('/clients')
def get_clients():
    conn = sqlite3.connect('data_base.db')
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    check_table_query = '''
        SELECT * FROM User
    '''
    cursor.execute(check_table_query)
    clients = cursor.fetchall()
    return render_template('Liste_Clients.html', tous_les_clients=clients)

@app.route('/formulaire_news')
def form_news():
    return render_template("search_company.html")

@app.route('/fetch_news', methods=['POST'])
def fetch_news():
    user = request.form.get('user')
    ticker = request.form.get('ticker')

    # Assurez-vous de mettre correctement votre clé d'autorisation pour l'API devapi.ai
    headers = {
        'Authorization': f'Bearer {os.getenv("API_KEY")}'
    }
    params = {
    'ticker': f'{ticker}',
    }
    # Récupération des actualités de l'API devapi.ai
    response = requests.get(f'https://devapi.ai/api/v1/markets/news', headers=headers,params=params)

    if response.status_code == 200:
        news_data = response.json()

        # Enregistrement des logs dans la base de données
        conn = sqlite3.connect('data_base.db')
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO logs (user, datetime, ticker)
            VALUES (?, ?, ?)
        ''', (user, datetime.now().isoformat(), ticker))
        conn.commit()
        conn.close()

        formatted_news = []
        for item in news_data['body']:
            formatted_news.append({
                'title': item['title'],
                'description': item['description'],
                'link': item['link'],
                'pubDate': item['pubDate']
            })

        return render_template('news_display.html', news=formatted_news)

    else:
        return f'Erreur lors de la récupération des actualités + {headers} + {params}', 500

@app.route('/upload_view')
def upload_view():
    return render_template("upload.html")

@app.route('/upload_base', methods=['POST'])
def upload_base():
    if 'file' not in request.files:
        return 'Aucun fichier trouvé !'

    file = request.files['file']
    
    if file.filename == '':
        return 'Aucun fichier sélectionné !'

    if file:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            return 'Format de fichier non pris en charge !'

        # Calculer des statistiques de base (par exemple : moyenne, médiane, etc.)
        summary_stats = df.describe()

        return render_template('Stats.html', tables=[summary_stats.to_html(classes='data')])

@app.route('/stock_form')
def stock_form():
    return render_template("stock_form.html")
@app.route('/plot', methods=['POST'])
def plot():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Récupération des données sur le prix du stock avec yfinance
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Création du graphique
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Close'], label='Prix')
    plt.title(f"Prix du stock de {ticker} dans le temps")
    plt.xlabel('Date')
    plt.ylabel('Prix (en $)')
    plt.legend()
    
    # Sauvegarde du graphique dans un objet BytesIO
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    return render_template('stock_plot.html', plot_url=plot_url)

@app.route('/img_form')
def img_form():
    return render_template('upload_image.html')

@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return 'Error: Model not loaded.'
    
    if 'file' not in request.files:
        return 'Aucun fichier trouvé !'

    file = request.files['file']
    
    if file.filename == '':
        return 'Aucun fichier sélectionné !'


    img = Image.open(file)
    img = img.resize((28, 28)) 
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0 

    # Faire la prédiction avec le modèle
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return f"La prédiction du modèle est : {predicted_class}"


@app.route('/predict_draw_view')
def predict_draw_view():
    return render_template('draw_predict.html')

@app.route('/predict_draw', methods=['POST'])
def predict_draw():
    data = request.get_json()
    image_data = re.sub('^data:image/.+;base64,', '', data['image_data'])

    img = Image.open(BytesIO(base64.b64decode(image_data)))
    img = img.convert('L')  
    img = img.resize((28, 28))

    img = ImageOps.invert(img)  
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    return jsonify({'prediction': str(predicted_digit)})



if __name__ == '__main__':
    app.run(debug=True)




























