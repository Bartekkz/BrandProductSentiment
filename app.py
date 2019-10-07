#!/usr/bin/env python3

from flask import Flask, render_template, request, redirect, url_for
import warnings
import pandas as pd 
warnings.filterwarnings('ignore')

app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('landingPage.html')

@app.route('/end')
def end():
    return render_template('end.html')


@app.route('/analyze')
def analyze():
    return render_template('analyze.html', error='')


@app.route('/test')
def test():
    return render_template('test.html')


@app.route('/read', methods=['POST', 'GET'])    
def read_csv():
    print('reading...')
    f = request.files.get('data_file')
    try:
        data = pd.read_csv(f)
        print(type(data))
        print(data.head())
        return render_template('end.html', data=data.value[1]) 
    except:
        return render_template('analyze.html', error='Load csv file!')


if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5002)

'''
TODO:
    - finish route for predicting from csv file
    - create test analyzer route
    - add navbar
    - change text of error to sth like 'do not forget to upload csv file :)'
'''
