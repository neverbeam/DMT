import csv

infile = 'training_set_VU_DM_2014.csv'
outfile = 'training_set_VU_DM_2014_medium.csv'

with open(infile) as f, open(outfile, 'w') as o:
    reader = csv.reader(f)
    writer = csv.writer(o, delimiter=',') # adjust as necessary
    i = 0
    for row in reader:
        if i < 100000:
            writer.writerow(row)
        else:
            break
        i += 1