import csv

with open('ODI-2018.csv') as data:
    reader = csv.reader(data, delimiter = ",")

    # Skip the header lines
    for i in range(2):
        next(reader, None)

    tot = 0
    for row in reader:
        tot += 1
        print (row)

    #Standard info
    print("Amount of records: ", tot)
    print("Amount of attributes:", len(row))
