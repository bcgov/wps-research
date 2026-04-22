'''
Takes a file directory containing a composite sequence and plots the percent of data in the frame vs date of the frame.
'''
from misc import extract_date, read_binary
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import numpy as np
from datetime import datetime

def extract_data_percent(file_dir, date=None):
    '''
    extracts and plots data percentage vs time
    '''
    #reading directory and sorting bin files
    files = os.listdir(file_dir)
    file_list = []
    for n in range(len(files)):
        if files[n].split('.')[-1] == 'bin':
            file_list.append(files[n])
        else:
            continue
    sorted_file_names = sorted(file_list, key=extract_date) 

    #calculating percent data for each frame
    percent_data = []
    dates = []
    ticks = []
    i = 0
    for file in sorted_file_names:
        data = read_binary(f'{file_dir}/{file}')[3]
        nans = sum(math.isnan(x) for x in data)
        percent_data.append(100 - 100*nans/len(data))
        dates.append(datetime.strptime(extract_date(file), '%Y%m%d').date())
        if percent_data[-1] >= 99.9:
            break
        print(percent_data[-1])
        if i % 10 == 0: 
            ticks.append(extract_date(file))
        i += 1

    #plotting BARC start date if provided 
    plt.figure()
    if date != None:
        date = datetime.strptime(str(date), '%Y%m%d').date()
        yvals = np.linspace(0,100,2)
        xvals = [date,date]
        for i in range(len(dates)):
            if dates[i] == date:
                percent = percent_data[i]
                plt.plot(xvals,yvals,color='red', label=f'BARC start date: data={round(percent,1)}%')
    plt.title('Percent data vs time of composite image sequence')
    plt.xlabel('Date')
    plt.ylabel('% data')
    plt.xticks(rotation=45)
    plt.plot(dates, percent_data)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig(f'{file_dir}_percent_data.png')