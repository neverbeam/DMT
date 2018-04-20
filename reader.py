import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score



def time(value):
    """ Parsing the time field."""
    date, time = value.split(" ")
    day, month, year = map(int, date.split("/"))
    hour, min, sec = map(int, time.split(":"))
    return datetime.datetime(year, month, day, hour, min, sec)

def programme(value):
    """ Parsing of the programme field."""
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
#    for math_string in ["math"]:
#        if math_string in value:
#            course = "MATH"
#    for phy_string in ["physics"]:
#        if phy_string in value:
#            course = "PHY"
    for eco_string in ["econo"]:
        if eco_string in value:
            course = "ECO"
    for phd_string in ["phd"]:
        if phd_string in value:
            course = "PhD"
    return course

def information_retrieval(value):
    pass

def machine_learning(value):
    pass

def statistics_course(value):
    pass

def database_course(value):
    pass

def gender(value):
    pass

def chocolate(value):
    pass

def birthday(value):
    """Parsing the birthday in the right format"""
    months = [["jan"],["feb"],["mar","maa"],["apr"],["may","mei"],["jun"],
             ["jul"],["aug"],["sep"],["oct","okt"],["nov"],["dec"]]

    for delimiter in ["/","-","."," "]:
        try:
            entry_1, entry_2, entry_3 = value.split(delimiter)


            # Indentify year.
            if int(entry_1) > 1960 and int(entry_1) < 2000:
                year = int(entry_1)
                day = int(entry_3)
            elif int(entry_3) > 1960 and int(entry_3) < 2000:
                year = int(entry_3)
                day = int(entry_1)

            # Parse month.
            for i in range(len(months)):
                for alt in months[i]:
                    if alt in entry_2.lower():
                        return datetime.date(year, i+1, day)
            month = int(entry_2)

            return datetime.date(year, month, day)

        except:
            pass
    return None

def neighbors(value):
    """Strips the random numbers into domain [0,8] or None otherwise"""
    try:
        number = int(value)
        if number <= 8 and number >= 0:
            return number
        else:
            return None
    except ValueError:
        return None

def standup(value):
    pass

def money(value):
    pass

def random_num(value):
    """Strips the random numbers into domain [0,10] or None otherwise."""
    try:
        number = int(value)
        if number <= 10 and number >= 0:
            return number
        else:
            return None
    except ValueError:
        return None

def bedtime(value):
    """ Parse the bedtime in day month year format. """
    day, month, year = 4, 5, 2018
    if value.isdigit():
        if len(value) == 4:
            hours = int(value[0:2])
            mins = int(value[2:4])         
        elif len(value) <= 2:
            hours = int(value)%24
            mins = 0
        else: return None
        if hours >= 6 and hours < 12: day, hours = day-1, hours+12
        elif hours >= 12 and hours < 18: hours = (hours+12)%24
        elif hours >= 18: day = day-1
        return datetime.datetime(year, month, day, hours, mins)
    try:
        entry_1, entry_2 = value.split(":")
        hours = int(entry_1)%24
        mins = int(''.join([i for i in entry_2 if i.isdigit()]))
        if hours > 6 and hours <= 12: day, hours = day-1, hours+12
        elif hours >= 12 and hours < 18: hours = (hours+12)%24
        elif hours >= 18: day = day-1
        return datetime.datetime(year, month, day, hours, mins)
    except:
        pass
    try:
        entry_1, entry_2 = value.split(".")
        hours = int(entry_1)%24
        mins = ''.join([i for i in entry_2 if i.isdigit()])
        mins = int(mins) if entry_2[0] != '0' else int(mins)*10
        if hours > 6 and hours <= 12: day, hours = day-1, hours+12
        elif hours >= 12 and hours < 18: hours = (hours+12)%24
        elif hours >= 18: day = day-1
        return datetime.datetime(year, month, day, hours, mins)
    except:
        pass
    try:
        entry_1, entry_2 = value.split(" ")
        if entry_1.isdigit():
            hours = int(entry_1)%24
            mins = int(entry_2) if entry_2.isdigit() else 0
            if hours > 6 and hours <= 12: day, hours = day-1, hours+12
            elif hours >= 12 and hours < 18: hours = (hours+12)%24
            elif hours >= 18: day = day-1
            return datetime.datetime(year, month, day, hours, mins)
    except:
        pass
    return None

def goodday1(value):
    pass

def goodday2(value):
    pass


def parse_data(data):
    """ Call a parsing function whenever the column needs parsing."""

    # Parse time
    data.time = data.time.apply(time)
    
    # Parse bed-time
    data.bedtime = data.bedtime.apply(bedtime)
    
    # Parse birthday
    data.birthday = data.birthday.apply(birthday)

    # Parse programme
    data["programme"] = data["programme"].apply(programme)

    # Random number stripping
    data["random_num"] = data["random_num"].apply(random_num)

    # Parse neighbors
    data["neighbors"] = data["neighbors"].apply(neighbors)



def plot_stats(data):
    """Make a distribution plot for every attribute"""
    # Plot programme.
    plt.hist(data["programme"], bins=range(len(np.unique(data["programme"]))+1), align="left", rwidth=0.8)
    plt.xlabel("programme")
    plt.ylabel("count")
    plt.show()

    # Plot random number.
    plt.hist(data[data["random_num"].notnull()]["random_num"], bins=range(12), align='left', rwidth=0.8)
    plt.xlabel("random number")
    plt.ylabel("count")
    plt.show()

    # Plot neighbors.
    plt.hist(data[data["neighbors"].notnull()]["neighbors"], bins=range(10), align='left', rwidth=0.8)
    plt.xlabel("num neighbors")
    plt.ylabel("count")
    plt.show()

    # Plot not null machine learning.
    values = data.machine_learning.value_counts()
    labels, sizes = ["yes","no","?"], values
    plt.pie(sizes, labels=labels,shadow=True, startangle=0, wedgeprops={"edgecolor":"k",'linewidth': 1})
    plt.rcParams['lines.linewidth'] = 2
    plt.title("Machine learning experience")
    plt.show()
    
    # Plot statistic experience.
    values = data.statistics_course.value_counts()
    labels, sizes = ["yes","no","?"], values
    plt.pie(sizes, labels=labels,shadow=True, startangle=0, wedgeprops={"edgecolor":"k",'linewidth': 1})
    plt.title("Statistics experience")
    plt.show()
    
    # Plot information course experience.
    values = data.information_retrieval.value_counts()
    labels, sizes = ["yes","no","?"], values
    plt.pie(sizes, labels=labels,shadow=True, startangle=0, wedgeprops={"edgecolor":"k",'linewidth': 1})
    plt.title("Information retrieval experience")
    plt.show()

    # Plot databases experience.
    values = data.database_course.value_counts()
    labels, sizes = ["yes","no","?"], values
    plt.pie(sizes, labels=labels,shadow=True, startangle=0, wedgeprops={"edgecolor":"k",'linewidth': 1})
    plt.title("Databases experience")
    plt.show()

    # Plot gender.
    values = data.gender.value_counts()
    labels, sizes = ["male","female","?"], values
    plt.pie(sizes, labels=labels,shadow=True, startangle=0, wedgeprops={"edgecolor":"k",'linewidth': 1})
    plt.title("Gender")
    plt.show()
    
    # Plot bedtime.
    dfb = pd.DataFrame({'bedtime':pd.to_datetime(data.bedtime)})
    dfb.set_index('bedtime', drop=False, inplace=True)
    vals = dfb.groupby(pd.Grouper(freq='60Min')).count()
    hours = [int(pd.to_datetime(d).strftime('%H')) for d in vals.index.values]
    labels = [str(h) + '-'+ str((h+1)%24) for h in hours]
    counts = [int(i) for i in vals.bedtime]
    plt.bar(range(11), counts)
    plt.xticks(range(11), labels, rotation='30')
    plt.xlabel('bedtime')
    plt.ylabel('count')
    plt.show()

    # Plot Choco evaluation.
    plt.hist(data[data["chocolate"].notnull()]["chocolate"], bins=range(6), align='left')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=65)
    plt.xlabel("Chocolate evaltuion")
    plt.ylabel("count")
    plt.show()
    
    # Plot standup.
    values = data.standup.value_counts()
    labels, sizes = ["no","yes","?"], values
    plt.pie(sizes, labels=labels,shadow=True, startangle=0, wedgeprops={"edgecolor":"k",'linewidth': 1})
    plt.title("Did you stand up?")
    plt.show()


def split_data(data, p=0.5):
    """Split the data into a learn and test set."""
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    split_at = int(p*len(data))
    learn = shuffled_data[:split_at].reset_index(drop=True)
    test = shuffled_data[split_at:].reset_index(drop=True)
    return learn, test


def learn_programme(learn_data):
    """Train naive bayes to predict the programme based on previous courses."""
    # Transformation of strings to number for learning.
    encoder = preprocessing.LabelEncoder()
    a = np.ravel(learn_data[["machine_learning", "information_retrieval", "statistics_course", "database_course", "programme"]])
    encoder.fit_transform(a)

    # Use X as the predictors and Y as what to predict.
    X = learn_data[["machine_learning", "information_retrieval", "statistics_course", "database_course"]].apply(encoder.transform)
    Y = encoder.transform(learn_data["programme"])
    learner = GaussianNB()
    learner.fit(X,Y)
      
    return learner, encoder
    
def predict_programme(predict_data, learner, encoder):
    """ Use naive bayes to predict the programme based on previous courses."""
    #Encode the predictons 
    encoded_data = predict_data[["machine_learning", "information_retrieval", "statistics_course", "database_course"]].apply(encoder.transform)
    
    prediction = learner.predict(encoded_data)
    prediction = encoder.inverse_transform(prediction)
        
    return prediction

def compare(data, prediction):
    """Compare the prediction and the actual data.""" 
    # Explicitly cast the datatypes to numpy arrays.
    counter = np.sum(np.array(data) == np.array(prediction))
            
    print("Number of correct guesses", counter)

def create_tree(learn_data):
    """ Create a decision tree for predicting programme based on prevous courses. """
    grandtree = DecisionTreeClassifier()
    
    # Transformation of strings to number for learning.
    encoder = preprocessing.LabelEncoder()
    a = np.ravel(learn_data[["machine_learning", "information_retrieval", "statistics_course", "database_course", "programme"]])
    encoder.fit_transform(a)
    
    # Use X as the predictors and Y as what to predict.
    X = learn_data[["machine_learning", "information_retrieval", "statistics_course", "database_course"]].apply(encoder.transform)
    Y = encoder.transform(learn_data["programme"])
    
    grandtree.fit(X, Y)
    return grandtree, encoder
    
def predict_programme_tree(test_data, tree, encoder):
    """ Use a decision tree for predicting programme based on prevous courses. """
    test_data = test_data[["machine_learning", "information_retrieval", "statistics_course", "database_course"]].apply(encoder.transform)
    
    prediction = tree.predict(test_data)
    prediction = encoder.inverse_transform(prediction)
    
    return prediction

def cross_validation(data):
    """ Use a decision tree and naive bayes to predict programme based on previous courses 
    with k-fold cross validation"""
    # Encode strings to int.
    encoder = preprocessing.LabelEncoder()
    encode_data = np.ravel(data[["machine_learning", "information_retrieval", "statistics_course", "database_course", "programme"]])
    encoder.fit_transform(encode_data)
    
    X = data[["machine_learning", "information_retrieval", "statistics_course", "database_course"]].apply(encoder.transform)    
    Y = encoder.transform(data["programme"])
    # Naive bayes to predict programme.
    learner = BernoulliNB()
    scores = cross_val_score(learner, X, Y, cv=4)
    print ("Accuracy Naive Bayes Bernoulli", scores.mean())
    learner = DecisionTreeClassifier(criterion = "entropy")
    # Decision tree to predict programme.
    scores = cross_val_score(learner, X, Y, cv=4)
    print ("Accuracy decision tree ", scores.mean())

if __name__ == '__main__':
    headers = ["time", "programme", "machine_learning", "information_retrieval",
    "statistics_course", "database_course", "gender", "chocolate", "birthday", "neighbors", "standup", "money", "random_num", "bedtime", "goodday1", "goodday2" ]
    data = pd.read_csv('ODI-2018.csv', delimiter = ',', names = headers, skiprows = 2)

    parse_data(data)
    
    print(data)

    #plot_stats(data)

    learn_data, test_data = split_data(data)

#    # Naive bayes
#    learner, encoder = learn_programme(learn_data)
#    prediction = predict_programme(test_data, learner, encoder)
#
#    # Decision tree
#    tree, encoder = create_tree(learn_data)
#    prediction = predict_programme_tree(test_data, tree, encoder)

    cross_validation(data)