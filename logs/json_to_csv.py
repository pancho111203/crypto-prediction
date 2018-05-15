import json
import csv

name = 'simple_cnn'
f = open('{}.json'.format(name))
data = json.load(f)
f.close()

f = open('{}.csv'.format(name), 'w')
csv_file = csv.writer(f)
csv_file.writerow(data[0].keys())
for item in data:
    csv_file.writerow(item.values())

f.close()