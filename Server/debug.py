import os
from flask import Flask, render_template

app = Flask(__name__, template_folder='C:/projects/Celebrity Classifier/Server/templates')

@app.route('/')
def index():
    print("Template folder:", app.template_folder)
    print("Current working directory:", os.getcwd())
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)


