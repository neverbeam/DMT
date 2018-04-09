import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def birthday(data):
    pass

def neighbors(data):
    pass

def standup(data):
    pass

def money(data):
    pass

def random_num(data):
    """Strips the random numbers into domain [0,10] or None otherwise"""
    try:
        number = int(data)
        if number <= 10 and number >= 0:
            return number
        else:
            return None
    except ValueError:
        return None

def bedtime(data):
    pass

def goodday(data1, data2):
    pass

headers = ["time", "programme", "machine_learning", "information_retreaval",
"statistics_course", "database_course", "gender", "chocolate", "birthday", "neighbors", "standup", "money", "random_num", "bedtime", "goodday1", "goodday2" ]
data = pd.read_csv('ODI-2018.csv', delimiter = ',', names = headers, skiprows = 2)

# Time stripping
data["random_num"] = data["random_num"].apply(random_num)

# time plot
plt.hist(data[data["random_num"].notnull()]["random_num"], align='left')
plt.xlabel("random number")
plt.ylabel("count")
plt.show()
