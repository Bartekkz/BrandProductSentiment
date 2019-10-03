#!/usr/bin/env python3

from flask import Flask, render_template, request
import warnings
import pandas as pd 
warnings.filterwarnings('ignore')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/read', methods=['POST', 'GET'])    
def read_csv():
    print('REAdinG')
    f = request.files.get('data_file')
    data = pd.read_csv(f)
    if not f:
        'No file'
    else:
        print(type(data))
        print(data.head())
        return render_template('ans.html')



if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5002)
