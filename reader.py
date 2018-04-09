import csv

with open('ODI-2018.csv') as data:
    reader = csv.reader(data, delimiter = ",")

    data = (list(reader)[2:])

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



    #Standard info
    print("Amount of records: ", len(data))
    print("Amount of attributes:", len(data[0]))
