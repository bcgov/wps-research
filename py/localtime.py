import os

def run(c):
    return os.popen(c).read().strip()

print("local time: yyyymmdd hh")
b= run('vancouver_hour=$(TZ="America/Vancouver" date +"%H"); echo "$vancouver_hour"')


a = run('vancouver_date=$(TZ="America/Vancouver" date +"%Y%m%d"); echo "$vancouver_date"')

print(a, b)
