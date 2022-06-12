from unicodedata import name
from flask import Flask,render_template,request,url_for
from flask_restful import Api,Resource
import attributeClusteringComplete as km

app = Flask(__name__)
api = Api(app)

#Design

@app.route('/Subject/<score>',methods=['GET'])
def search(score):
    b = km.km_Subject(score)
    return render_template('index.html',score=b)

@app.route('/Subject/<score>/<reqjob>',methods=['GET'])
def search2(score,reqjob):
    b,r = km.km_Subject(score)#Return Major,S
    j = km.km_JobReq(b,reqjob)#Return Job
   
    return render_template('index2.html',score=b,job=j,requireSubject=r)

@app.route('/')
def home():
    columnSubject = km.colum
    return render_template('home.html',columnSubject=columnSubject)



if __name__ == "__main__":
    app.run()