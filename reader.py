import pandas as pd
import datetime

def time(data):
    pass

def programme(data):
    pass

def information_retreaval(data):
    pass

def machine_learning(data):
    pass

def statistics_course(data):
    pass

def database_course(data):
    pass

def gender(data):
    pass

def chocolate(data):
    pass

def birthday(value):

    months = [["jan"],["feb"],["mar","maa"],["apr"],["may","mei"],["jun"],
             ["jul"],["aug"],["sep"],["oct","okt"],["nov"],["dec"]]

    for delimiter in ["/","-","."," "]:
        try:
            entry_1, entry_2, entry_3 = value.split(delimiter)
        
            
            # indentify year
            if int(entry_1) > 1960 and int(entry_1) < 2000:
                year = int(entry_1)
                day = int(entry_3)
            elif int(entry_3) > 1960 and int(entry_3) < 2000:
                year = int(entry_3)
                day = int(entry_1)
                
            # parse month
            for i in range(len(months)):
                for alt in months[i]:
                    if alt in entry_2.lower():
                        return datetime.date(year, i+1, day)
            month = int(entry_2)
            
            return datetime.date(year, month, day)

        except:
            pass
    return None

def neighbors(data):
    pass

def standup(data):
    pass

def money(data):
    pass

def random_num(data):
    pass

def bedtime(data):
    pass

def goodday(data1, data2):
    pass

headers = ["time", "programme", "machine_learning", "information_retreaval",
"statistics_course", "database_course", "gender", "chocolate", "birthday", "neighbors", "standup", "money", "random_num", "bedtime", "goodday1", "goodday2" ]
data = pd.read_csv('ODI-2018.csv', delimiter = ',', names = headers, skiprows = 2)

data.birthday = data.birthday.apply(birthday)