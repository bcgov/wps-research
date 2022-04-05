'''deprecated script: should modify this to put the data in a folder
..so we can see the scripts'''
import os
import sys
def err(m):
    print("Error: " + str(m)); sys.exit(1)

def run(c):
    print(c)
    a = os.system(c)
    if a != 0: err("command failed: " + c)

# step 1: list zipfiles not downloaded yet...
lines = [x.strip() for x in open("/home/" + os.popen('whoami').read().strip() + "/sentinel2/download.sh").readlines()]
print(len(lines))

zf = [line.split()[3] for line in lines]
for z in zf:
    if z[-3:] != "zip":
        err("not zip extension: " + z)

print(zf[-1])
pi = [z[:-4] for z in zf]
print(pi[-1])

print("read google data files..")
g_base = [x.strip() for x in open("index.csv_BASE_URL").readlines()]
if g_base[0] != "BASE_URL":
    err("expected: BASE_URL")
g_base = g_base[1:]

g_prod = [x.strip() for x in open("index.csv_PRODUCT_ID").readlines()]
if g_prod[0] != "PRODUCT_ID":
    err("expected: PRODUCT_ID")
g_prod = g_prod[1:]

print("creating lookup..")
g_prod_to_url = {g_prod[i]: g_base[i] for i in range(len(g_base))}

f = open("gcloud_download.sh", "wb")
g = open("gcloud_not_found.txt", "wb")
print("applying lookup..")

for p in pi:
    if p not in g_prod_to_url:
        g.write((p + "\n").encode())
    else:
        print(g_prod_to_url[p]) 
        f.write(("test ! -f " + p + ".SAFE && " +
                 "gsutil cp -r " + g_prod_to_url[p] + " ./ " +
                 "> " + p + ".SAFE_stdout.txt 2> " + p + ".SAFE_stderr.txt" + "\n").encode())
f.close()
g.close()


# step 2: cross-reference with index file..
# create a list of download commands e.g.:
# gsutil cp -r gs://gcp-public-data-sentinel-2/tiles/44/X/MQ/S2B_MSIL1C_20200526T101559_N0209_R065_T44XMQ_20200526T123451.SAFE ./
