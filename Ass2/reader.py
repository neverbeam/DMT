import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from ranking_measures import find_dcg, find_ndcg

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
        # data[category] = data[category].astype(str)
        if show:
            column_to_pie(data, category)

    # data["date_time"] = data["date_time"].apply(parse_date_time)

    return data

def stripcompetition(item):
    if np.isnan(item):
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
def get_single_test(test_data, srch_id=None):
    if srch_id == None:
        # get a random srch id
        srch_id = test_data.sample().iloc[0]['srch_id'] #just get that 1 value
    query_with_booking = get_single_query(test_data, srch_id)
    query_test = query_with_booking.drop(columns=["click_bool","gross_bookings_usd","booking_bool"])
    return query_test, srch_id

# given a prediction ordering of likeliness that hotel was booked,
# returns a value describing how good the prediction is by using nDCG
def try_single_test(test_data, srch_id, predictions):
    query_with_booking = get_single_query(test_data, srch_id)

    # get the real scores for the predicted ranking of the hotels
    hypothesis = []
    for i in range(len(predictions)):
        real_value = 0
        clicked = query_with_booking.iloc[i]['click_bool']
        if clicked == 1:
            booked = query_with_booking.iloc[i]['booking_bool']
            if booked == 1:
                real_value = 5
            else:
                real_value = 1
        hypothesis.append(real_value)

    # the best order is just the highest ranked on top
    reference = list(reversed(sorted(hypothesis)))

    # get the normalized discounted cumulative gain
    score = find_ndcg(reference, hypothesis)
    return score

    
# applies a ranker on a test set and returns its score
def test_ranker(test_data, ranker):
    # loop over all queries in test set
    all_test_srch_ids = pd.unique(test_data['srch_id'])
    results = []
    for srch_id in all_test_srch_ids:
        # get a single test query
        query_test, _ = get_single_test(test_data, srch_id)

        # predict the order of the prop_id (for now hardcoded list of numbers to test)
        predictions = list(query_test["prop_id"]) # <- pls order this

        # test your prediction
        result = try_single_test(test_data, srch_id, predictions)

        results.append(result)
        
    return sum(results)/len(results)


if __name__ == '__main__':
    train_data = pd.read_csv('training_set_VU_DM_2014_small.csv', delimiter = ',')
    test_data = pd.read_csv('test_set_VU_DM_2014_small.csv', delimiter = ',')

    #print (train_data)

    # all column headers
    categories = list(train_data)
    # show em
    train_data = parse_data(train_data, categories, False)

    train_data = parse_data(train_data, categories)
    test_data = parse_data(test_data, categories)

    #train_data = changecompetition(train_data, categories)
    #print (train_data["comp2_rate"])
    
#    correlations = train_data[2:23].corr()
#    # look at the query range
#    for i in range(2,24):
#        print("\n===============" + categories[i])
#        print(train_data[2:23].corr()[categories[i]])



    # LEARNING, PREDICTING, SCORING
    # split the data
    #learn_data, test_data = split_data(train_data)

    # do some learn step here
    ranker = None
    
    # get a single test query
#    query_test, srch_id = get_single_test(test_data)
#
#    # predict the order of the prop_id (for now hardcoded list of numbers to test)
#    predictions = list(query_test["prop_id"]) # <- pls order this
#
#    # test your prediction
#    result = try_single_test(test_data, srch_id, predictions)
#
#    print(result)
    
    
    import pyltr
    
    metric = pyltr.metrics.NDCG(k=10)
    
    model = pyltr.models.LambdaMART(
    metric=metric,
    n_estimators=1000,
    learning_rate=0.02,
    max_features=0.5,
    query_subsample=0.5,
    max_leaf_nodes=10,
    min_samples_leaf=64,
    verbose=1,
    )

    ids = train_data["srch_id"]
    TX = train_data[["prop_starrating", "prop_location_score1",  "price_usd"]]
    TY = 5 * train_data["booking_bool"] + train_data["click_bool"]
    model.fit(TX, TY, ids)
    
    
    ids = test_data["srch_id"]
    EX =  test_data[["prop_starrating", "prop_location_score1",  "price_usd"]]
    EY = 5 * test_data["booking_bool"] + test_data["click_bool"]
    Epred = model.predict(EX)
    #print ('Random ranking:', metric.calc_mean_random(ids, EY))
    #print ('Our model:', metric.calc_mean(ids, EY, Epred))

    test_result = test_ranker(test_data, ranker)
    print(test_result)
