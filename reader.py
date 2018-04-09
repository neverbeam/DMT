import pandas as pd
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

def birthday(data):
    pass

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

# parse programme
data["programme-parsed"] = data["programme"].apply(programme)
print(data[["programme","programme-parsed"]].to_string())
plt.hist(data["programme-parsed"], bins=range(len(np.unique(data["programme-parsed"]))+1), align="left", rwidth=0.8)
plt.show()