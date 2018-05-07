import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime
from sklearn import preprocessing

def parse_date_time(value):
    new_val = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
    return new_val

def column_to_pie(data, category):
    values = data[category].value_counts()
    labels, sizes = values.index.values, values
    plt.pie(sizes, labels=labels)
    plt.title(category)
    plt.show()

def parse_data(data, categories, show=False):
    for category in categories:
        # count NaNs
        num_nan = len([d for d in data[category] if math.isnan(d)]) if isinstance(data[category][0], float) else 0
        if isinstance(data[category][0],float):
            # set all NaNs to mean
            data[category][np.isnan(data[category])] = np.mean(data[category])
            # normalize
            data[category] = preprocessing.normalize(data[category].reshape(1,-1),norm='l2').ravel()
        if show:
            column_to_pie(data, category)
    data["date_time"] = data["date_time"].apply(parse_date_time)

    return data

def stripcompetition(item):
    if item == "nan":
        return 0
    else:
        return item
    
def changecompetition(data, categories):
    for category in categories[-24:-3]:
        data[category] = data[category].apply(stripcompetition)
        
    return data

# split the data in a learn and test set based on search ids
def split_data(data, p=0.5):
    """Split the data into a learn and test set."""
    # get all different search ids and shuffle them
    srch_ids = data["srch_id"].unique()
    np.random.shuffle(srch_ids)

    # split the ids into a learn and test set
    split_at = int(p*len(srch_ids))
    learn = data.loc[data['srch_id'].isin(srch_ids[:split_at])]
    test = data.loc[data['srch_id'].isin(srch_ids[split_at:])]

    return learn, test

# returns rows where the search id is the one given
def get_single_query(data, srch_id):
    return data.loc[data['srch_id'] == srch_id]

# returns the rows without bookings of a randomly picked query_id
def get_single_test(test_data):
    srch_id = test_data.sample().iloc[0]['srch_id'] #just get that 1 value
    query_with_booking = get_single_query(test_data, srch_id)
    query_test = query_with_booking.drop(columns=["click_bool","gross_bookings_usd","booking_bool"])
    return query_test, srch_id

# given a prediction ordering of likeliness that hotel was booked,
# returns a value describing how good the prediction is
# voor nu nog: pakt de top 5 en kijkt of die geboekt zijn en geeft er punten voor
# later moet dit nog worden gedaan met cummulative gain etc
def try_single_test(test_data, srch_id, predictions):
    query_with_booking = get_single_query(test_data, srch_id)
    score = 0
    for i in range(5):
        prediction = predictions[i]
        predicted_row = query_with_booking.loc[query_with_booking['prop_id'] == prediction]
        if predicted_row.iloc[0]['click_bool'] == 1:
            if predicted_row.iloc[0]['booking_bool'] == 1:
                score += 5
            else:
                score += 1

    return score

if __name__ == '__main__':
    train_data = pd.read_csv('training_set_VU_DM_2014_small.csv', delimiter = ',')

    # all column headers
    categories = list(train_data)
    # show em
    #train_data = parse_data(train_data, categories, True)

    train_data = parse_data(train_data, categories)

    train_data = changecompetition(train_data, categories)
    # print (train_data["comp2_rate"])
    
    # correlations = train_data[2:23].corr()
    # # look at the query range
    # for i in range(2,24):
    #     print("\n===============" + categories[i])
    #     print(train_data[2:23].corr()[categories[i]])



    # LEARNING, PREDICTING, SCORING
    # split the data
    learn_data, test_data = split_data(train_data)

    # do some learn step here
    
    # get a single test query
    query_test, srch_id = get_single_test(test_data)

    # predict the order of the prop_id (for now hardcoded list of numbers to test)
    predictions = list(query_test["prop_id"]) # <- pls order this

    # test your prediction
    result = try_single_test(test_data, srch_id, predictions)

    print(result)