import mysql.connector

host = '127.0.0.1'
username = 'root'
password = ''

class Connection:
    def __init__(self, host, username,password):
        self.host = host
        self.username = username
        self.password = password
    
    def connectToSql(self,table):
        mydb = mysql.connector.connect(
        host = self.host,
        user = self.username,
        password = self.password,
        database = 'cos4501'
        )
        myCursor = mydb.cursor()
        myCursor.execute("SELECT * FROM " + table)
        myresult = myCursor.fetchall()
        return myresult


# TODO
conn = Connection(host,username,password).connectToSql('dataservey')
print(type(conn))

