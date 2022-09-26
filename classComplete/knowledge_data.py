from tkinter.tix import COLUMN
import pandas as pd
import connection

host = '127.0.0.1'
username = 'root'
password = ''

class KnowLedge:
    column = ['สถิติ', 'การแจกแจงความน่าจะเป็นเบื้องต้น', 'ลำดับและอนุกรม',
    'แคลคูลัส', 'เรขาคณิตวิเคราะห์', 'เซต', 'ตรรกศาสตร์',
    'จำนวนจริงและพหุนาม', 'ฟังก์ชัน', 'ฟังก์ชันตรีโกณมิติ', 'จำนวนเชิงซ้อน',
    'เมทริกซ์', 'เวกเตอร์ในสามมิติ', 'หลักการนับเบื้องต้น', 'ความน่าจะเป็น',
    'ฟิสิกส์', 'เคมี', 'ชีวะ','Y']

    def __init__(self,tableName):
        listItem = connection.Connection(host,username,password).connectToSql(tableName)
        self.df = pd.DataFrame(data=listItem,columns=self.column)

class Skill:
    column =['Coding','Programming','Artificial Intelligence','Project','Business','Engineer','Y']
    def __init__(self):
        listItem = connection.Connection(host,username,password).connectToSql('skill_major')
        self.df = pd.DataFrame(data=listItem,columns=self.column)

# df = Skill()
# print(df.df)

