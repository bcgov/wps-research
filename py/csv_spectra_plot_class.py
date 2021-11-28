'''20211128 plot spectra, labelled by class
input: csv
  * spectra fields are identified by ending in nm
input: field label of class..
...spectra with the same label are given the same color and same key in legend'''
import os
import sys
import csv


def read_csv(f):
    f, data, i = 'survey_label.csv', [], 0
    reader = csv.reader(open(f), delimiter=',', quotechar='"')

    for row in reader:
        row = [x.strip() for x in row]
        if i == 0:
            N = len(row)
            I = range(N)
            fields, data = row, [[] for j in I]
        else:
            for j in I:
                data[j].append(row[j])
        i += 1
        if i % 1000 == 0:
            print(i)

    fields = [x.strip().replace(' ', '_') for x in fields] # spaces are always bad!
    return data





