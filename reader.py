import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

def time(data):
    pass

def programme(value):
    value = value.lower()
    course = "Other"
    for business_string in ["b"]:
        if business_string in value:
            course = "BUS"
    for cs_string in ["cs", "computer"]:
        if cs_string in value:
            course = "CS"
    for cls_string in ["cls", "computational"]:
        if cls_string in value:
            course = "CLS"
    for ai_string in ["ai", "artificial", "intelligence"]:
        if ai_string in value:
            course = "AI"
    for business_string in ["business"]:
        if business_string in value:
            course = "BUS"
    for ba_string in ["ba","business analytics"]:
        if ba_string in value:
            course = "BA"
    for bio_string in ["bio"]:
        if bio_string in value:
            course = "BIO"
    for math_string in ["math"]:
        if math_string in value:
            course = "MATH"
    for phy_string in ["physics"]:
        if phy_string in value:
            course = "PHY"
    for eco_string in ["econo"]:
        if eco_string in value:
            course = "ECO"
    for phd_string in ["phd"]:
        if phd_string in value:
            course = "PhD"
    return course

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

data.birthday = data.birthday.apply(birthday)

# parse programme
data["programme"] = data["programme"].apply(programme)
# programme plot
plt.hist(data["programme"], bins=range(len(np.unique(data["programme"]))+1), align="left", rwidth=0.8)
plt.xlabel("programme")
plt.ylabel("count")
plt.show()

# Time stripping
data["random_num"] = data["random_num"].apply(random_num)
# time plot
plt.hist(data[data["random_num"].notnull()]["random_num"], bins=range(12), align='left', rwidth=0.8)
plt.xlabel("random number")
plt.ylabel("count")
plt.show()
