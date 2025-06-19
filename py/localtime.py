import os

def run(c):
    return os.popen(c).read().strip()

print("local physical time: yyyymmdd hh (Vancouver time zone)")
b= run('vancouver_hour=$(TZ="America/Vancouver" date +"%H"); echo "$vancouver_hour"')


a = run('vancouver_date=$(TZ="America/Vancouver" date +"%Y%m%d"); echo "$vancouver_date"')

print(a, b)

print("configured time: yyyymmdd hh")
b = run('local_hour=$(date +"%H"); echo "$local_hour"')
a = run('local_date=$(date +"%Y%m%d"); echo "$local_date"')
print(a,b)

