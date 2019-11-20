# -------------------------------------------------------------------------
import os
import sys
print "species,dominant"
lines = open("LINE_3_TRE.lut").readlines() # read the lines from the file
for line in lines: # go through the lines in the file
    chunks = line.strip().split(",") # the file is a CSV, split on comma
    s = chunks[0]
    s = s.strip() # remove whitespace
    
    if s == "": # if the string is empty: "empty cell"
        continue  # go to the next line
    
    if not s[0].isupper():
        print "Error"
        sys.exit(1)

    s = s.split("(")[0] # get rid of everything after and including the bracket

    i = 1 # start the count at the next letter (0 is the first index)
    
    #print len(s), [s]
    
    while i < len(s) and (not s[i].isupper()):
        i += 1

    print chunks[0] + "," + s[0:i]
