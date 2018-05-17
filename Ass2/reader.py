import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from datetime import datetime
from sklearn import preprocessing
from ranking_measures import find_dcg, find_ndcg
import math
from sklearn import preprocessing
import pyltr
import pickle
import time

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
    # add features:
    star_diff = [(np.log(i.price_usd) - i.prop_log_historical_price ) if i.price_usd > 0 
                and i.prop_log_historical_price > 0 else 0 for j,i in data.iterrows()]
    data['star_diff'] = pd.Series(star_diff)
    
    price_diff = [(i.prop_starrating - i.visitor_hist_starrating ) if i.prop_starrating > 0 
                and i.visitor_hist_starrating > 0 else 0 for j,i in data.iterrows()]
    
    data['price_diff'] = pd.Series(price_diff)


    """ Parse all the data to set NaNs to mean. """
    for category in categories:
        # count NaNs
        if isinstance(data.iloc[0][category],float):
            # set all NaNs to mean
            feat_mean = np.mean(data[category])
            data[category] = data[category].fillna(feat_mean)
            num_nan = len([d for d in data[category] if math.isnan(d)])
            # normalize
            data[category] = preprocessing.normalize(data[category].values.reshape(1,-1),norm='l2').ravel()

            data = data.fillna(data.mean())

        if show:
            column_to_pie(data, category)
    return data


    
def plot_importance(data, category):
    if isinstance(data.iloc[0][category],float) or isinstance(data.iloc[0][category],np.int64):
        booked = np.mean([i[category] for j,i in data.iterrows() if i.booking_bool == 1])
        clicked = np.mean([i[category] for j,i in data.iterrows() if i.click_bool == 1])
        neither = np.mean([i[category] for j,i in data.iterrows() if i.booking_bool == 0 and i.click_bool == 0])
        plt.bar([1,2,3], [neither, clicked, booked], tick_label=['neither', 'clicked','booked'])
        plt.title(category)
        plt.show()


# def stripcompetition(item):
#     if np.isnan(item):
#         return 0
#     else:
#         return item
    
# def changecompetition(data, categories):
#     for category in categories[-24:-3]:
#         data[category] = data[category].apply(stripcompetition)

# adds new feature columns to a dataset
# returns the names of added columns
def add_features(data):
    new_columns = []

    # location times price
    data['loc_price'] = data['prop_location_score1'] * data['price_usd']
    new_columns.append('loc_price')
    # location times starrating
    data['loc_star'] = data['prop_location_score1'] * data['prop_starrating']
    new_columns.append('loc_star')
    # starrating times price
    data['star_price'] = data['price_usd'] * data['prop_starrating']
    new_columns.append('star_price')
    # starrating * price * location
    data['star_price_loc'] = data['price_usd'] * data['prop_starrating'] * data['loc_price']
    new_columns.append('star_price_loc')

    return new_columns

def normalize(data, categories):
    """Normalize all the column that contain floats. """
    for category in categories:
        if isinstance(data.iloc[0][category],float):
            data[category] = data[category]/ data[category].max()

        
    return data


def update_comprate(train_data):
    """Set any weird competator data to 0. """
    for i in range(1,9):
        a = ("comp{}_rate_percent_diff".format(i))
        train_data[train_data[a] > 300 ] = 0
    
    return train_data
    
def average_competition(data):
    """ Average all the competition data. """
    average = data["comp1_rate_percent_diff"]

    for i in range(2,9):
        a = ("comp{}_rate_percent_diff".format(i))
        b = ("comp{}_rate".format(i))
        # The rate (positive/negative) * difference
        average += data[a] * data[b]

    return average/8

def price_difference(data):
    """Create a column with difference between current and historical price. """
    price = np.log(data["price_usd"])
    return price - data["prop_log_historical_price"]

def split_data(data, p=0.5):
    """Split the data into a learn and test set."""
    # Get all different search ids and shuffle them.
    srch_ids = data["srch_id"].unique()
    np.random.shuffle(srch_ids)

    # split the ids into a learn and test set
    split_at = int(p*len(srch_ids))
    learn = data.loc[data['srch_id'].isin(srch_ids[:split_at])]
    test = data.loc[data['srch_id'].isin(srch_ids[split_at:])]

    return learn, test

def get_single_query(data, srch_id):
    """ returns rows where the search id is the one given"""
    return data.loc[data['srch_id'] == srch_id]

def get_single_test(test_data, srch_id=None):
    """returns the rows without bookings of a randomly picked query_id. """
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
        # get the query rows one by one in order of predictions
        predicted_row = query_with_booking.loc[query_with_booking['prop_id'] == predictions[i]]

        real_value = 0.
        clicked = predicted_row.iloc[0]['click_bool']
        if clicked == 1:
            booked = predicted_row.iloc[0]['booking_bool']
            if booked == 1:
                real_value = 5.
            else:
                real_value = 1.
        hypothesis.append(real_value)

    # the best order is just the highest ranked on top
    reference = list(reversed(sorted(hypothesis)))

    def dcg(rels):
        score = 0.0
        for order, rel in enumerate(rels, start=1):
            score += float(2**rel - 1)/math.log(order+1, 2)
        return score

    def ndcg(reference, hypothesis):
        return dcg(hypothesis)/dcg(reference)

    # get the normalized discounted cumulative gain
    score = ndcg(reference, hypothesis)
    return score

    
# applies a ranker on a test set and returns its score
def test_ranker(test_data, model, cats):
    # loop over all queries in test set
    all_test_srch_ids = pd.unique(test_data['srch_id'])

    # or single value for error checking
    # random_query, random_srch_id = get_single_test(test_data)
    # all_test_srch_ids = [random_srch_id]

    results = []
    random_results = []
    for srch_id in all_test_srch_ids:
        # get a single test query
        query_test, _ = get_single_test(test_data, srch_id)
        prop_ids_unsorted = list(query_test["prop_id"])

        # this gives a ranking number to each query+hotel row in the dataset
        predict_result_unsorted = model.predict(query_test.as_matrix(cats))

        # if a list of unsorted predictions is given (meaning each row has a value of how good it is)
        # here we sort the rows and return their property ids in the correct order
        predict_result_rev_sorted, prop_ids_rev_sorted = zip(*sorted(zip(predict_result_unsorted, prop_ids_unsorted)))
        predict_result_sorted = list(reversed(predict_result_rev_sorted))
        prop_ids_sorted = list(reversed(prop_ids_rev_sorted))
        predictions = prop_ids_sorted

        # test your prediction
        result = try_single_test(test_data, srch_id, predictions)
        random_result = try_single_test(test_data, srch_id, prop_ids_unsorted)

        results.append(result)
        random_results.append(random_result)
        
    return sum(results)/len(results), sum(random_results)/len(random_results)


if __name__ == '__main__':
    train_data = pd.read_csv('training_set_VU_DM_2014_medium.csv', delimiter = ',')
    # test_data = pd.read_csv('test_set_VU_DM_2014_small.csv', delimiter = ',')

    # all column headers
    categories = list(train_data)

    # show em
    train_data = parse_data(train_data, categories, False)

    
    
    #train_data = changecompetition(train_data, categories)

    # add some additional features
    new_features = add_features(train_data)
    train_data = update_comprate(train_data)
    train_data = normalize(train_data, categories)
    train_data["comp_average"] = average_competition(train_data)
    train_data["price_diff"] = price_difference(train_data)


    # LEARNING, PREDICTING, SCORING
    # split the data into 75% train, 25% test
    learn_data, test_data = split_data(train_data, 0.75)

    # do some learn step here
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

    # set the categories

    LM_cats = ["star_diff", "prop_location_score2",  "price_diff", "promotion_flag", "random_bool"]
    #LM_cats = ["prop_starrating", "prop_location_score2",  "price_usd", "promotion_flag"]

    all_usable_cats = ['site_id', 'prop_starrating', 'prop_review_score', 
            'prop_brand_bool', 'prop_location_score1', 'prop_location_score2', 
            'prop_log_historical_price', 'price_usd', 'promotion_flag', 
            'srch_length_of_stay', 'srch_booking_window', 'srch_adults_count', 
            'srch_children_count', 'srch_room_count', 'srch_saturday_night_bool', 
            'orig_destination_distance', 'random_bool']

    LM_cats = ["prop_starrating", "prop_location_score1",  "price_usd"]


    ids = train_data.as_matrix(["srch_id"])[:,0] #maybe should be strings...
    TX = train_data.as_matrix(columns=all_usable_cats + new_features)
    # when both are true, score is 5, only click is 1, nothing is 0
    Ty = train_data["click_bool"] + 4*train_data["booking_bool"]
    # cast to float in order to do regression and not classification
    Ty = Ty.astype(float).as_matrix()

    # print('TX',TX.shape,TX)
    # print('Ty',Ty.shape,Ty)
    # print('ids',ids.shape,ids)
    modelname = 'LambdaMART_small_floats.sav'

    # fit and save the model
    model.fit(TX, Ty, ids)
    pickle.dump(model, open(modelname, 'wb'))

    modelname = 'LambdaMART_medium_all.sav'

    # fit and save the model
    # start_fit = time.time()
    # model.fit(TX, Ty, ids)
    # end_fit = time.time()
    # print('Time to fit:', (end_fit-start_fit)/60.)
    # pickle.dump(model, open(modelname, 'wb'))


    # load the model
    model = pickle.load(open(modelname, 'rb'))

    test_result, random_result = test_ranker(test_data, model, all_usable_cats)
    print('Random our ranking:', random_result)
    print('Model total', test_result)
