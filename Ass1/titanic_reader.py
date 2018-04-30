import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing

def age(value):
    bin_val = "unknown"
    if (0<value<14):
        bin_val = "child"
    elif (14<value<25):
        bin_val = "young adult"
    elif (25<value<40):
        bin_val = "adult"
    elif (40<value<60):
        bin_val = "old adult"
    elif (60<value):
        bin_val = "elderly"
    return bin_val

def fare(value):
    bin_val = "unknown"
    if (0<value<10):
        bin_val = "low"
    elif (10<value<30):
        bin_val = "medium"
    elif (30<value<100):
        bin_val = "high"
    elif (100<value):
        bin_val = "absurd"
    return bin_val

def column_to_pie(data, category):
    values = data[category].value_counts()
    labels, sizes = values.index.values, values
    plt.pie(sizes, labels=labels)
    # plt.title(category)
    plt.show()

def parse_data(data, categories, show=False):
    # parse age
    data["Age"] = data["Age"].apply(age)

    # parse fare
    data["Fare"] = data["Fare"].apply(fare)

    for category in categories:
        data[category] = data[category].astype(str)
        if show:
            column_to_pie(data, category)

    return data


# create the encoder for classification before splitting data,
# to make sure that all possible values are encoded
def create_encoder(data, categories):
    # Transformation of strings to number for learning.
    encoder = preprocessing.LabelEncoder()
    ravelled = np.ravel(data[categories+["Survived"]])
    encoder.fit_transform(ravelled)
    return encoder

# split the data into a learn and test set
def split_data(data, p=0.5):
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    split_at = int(p*len(data))
    learn = shuffled_data[:split_at].reset_index(drop=True)
    test = shuffled_data[split_at:].reset_index(drop=True)
    return learn, test

# use naive bayes to predict the programme based on previous courses
def learn_survive(learn_data, encoder, algorithm, categories):
    """Train naive bayes to predict the programme based on previous courses."""
    # Use X as the predictors and Y as what to predict.
    X = learn_data[categories].apply(encoder.transform)
    Y = encoder.transform(learn_data["Survived"])
    learner = algorithm()
    learner.fit(X,Y)
      
    return learner
    
def predict_survive(predict_data, learner, encoder, categories):
    """ Use naive bayes to predict the programme based on previous courses."""
    #Encode the predictons 
    encoded_data = predict_data[categories].apply(encoder.transform)
    
    prediction = learner.predict(encoded_data)
    prediction = encoder.inverse_transform(prediction)
        
    return prediction

def prediction_results(data, prediction):
    """Compare the prediction and the actual data.""" 
    # Explicitly cast the datatypes to numpy arrays.
    counter = np.sum(np.array(data) == np.array(prediction))
    return counter

def test_classifiers(data, encoder, categories, N):
    algorithms = [GaussianNB, BernoulliNB, SVC, RandomForestClassifier, DecisionTreeClassifier]
    algorithms_str = ["Gaussian Naive Bayes", "Bernoulli Naive Bayes", "Support Vector Machine",
                        "Random Forest", "Decision Tree"]

    algorithms_res = [[] for _ in range(len(algorithms))]
    for i in range(N):
        learn_data, test_data = split_data(data)

        # test the different classification algorithms
        for i in range(len(algorithms)):
            learner = learn_survive(learn_data, encoder, algorithms[i], categories)
            prediction = predict_survive(test_data, learner, encoder, categories)
            algorithms_res[i].append(prediction_results(test_data["Survived"], prediction))

    for i in range(len(algorithms)):
        print(algorithms_str[i] + " accuracy: " + str(sum(algorithms_res[i])/len(test_data)/N*100) + "%")

def test_categories(data, encoder, algorithm, categories, N, total):
    cats_res = [[] for _ in range(len(categories))]
    for j in range(N):
        learn_data, test_data = split_data(data)
        for i in range(len(categories)):
            # copy the categories
            del_cats = list(categories)

            # remove 1 category from the list
            category = categories[i]
            del del_cats[i]

            learner = learn_survive(learn_data, encoder, algorithm, del_cats)
            prediction = predict_survive(test_data, learner, encoder, del_cats)
            cats_res[i].append(prediction_results(test_data["Survived"], prediction))

    for i in range(len(categories)):
        print(categories[i] + " when removed: " + str(total - sum(cats_res[i])/len(test_data)/N*100) + "%")


if __name__ == '__main__':
    train_data = pd.read_csv('titanic_train.csv', delimiter = ',')
    test_data = pd.read_csv('titanic_test.csv', delimiter = ',')

    # list the categories we use to classify the passenger
    categories = ["Sex", "Age", "Pclass", "Fare", "SibSp", "Parch"]
    # write the data into correct format
    train_data = parse_data(train_data, categories + ["Survived"])
    test_data = parse_data(test_data, categories)
    # get the encoder for all the data (no missing values)
    encoder = create_encoder(train_data.append(test_data).fillna('1'), categories)

    # this checks whether NB or DT is better
    # test_classifiers(train_data, encoder, categories, 1000)
    # this checks which category is most influential
    # test_categories(train_data, encoder, SVC, categories, 1000, 80.)

    # train classifier on train and apply it to test
    # classifier = learn_survive(train_data, encoder, RandomForestClassifier, categories)
    # result = predict_survive(test_data, classifier, encoder, categories)
    # conclusion = test_data[["PassengerId"]]
    # conclusion["Survived"] = result
    # conclusion.to_csv(path_or_buf="titanic_survivors.csv", index=False)