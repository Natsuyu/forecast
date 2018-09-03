#! usr/bin/env python
#-*- coding:utf8 -*-

import csv
with open('/Volumes/张兆年/failure_prediction/bbu_data.csv') as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        print row



