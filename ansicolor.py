#!/usr/bin/env python
'''  ansi color codes for posix terminal'''

color = {"KNRM": "\\x1B[0m",  #normal
         "KBLK": "\\x1B[30m", #black
         "KRED": "\\x1B[31m", #red
         "KGRN": "\\x1B[32m", #green
         "KYEL": "\\x1B[33m", #yellow
         "KBLU": "\\x1B[34m", #blue
         "KMAG": "\\x1B[35m", #magenta
         "KCYN": "\\x1B[36m", #cyan
         "KWHT": "\\x1B[37m", #white
         "KBLD": "\\x1B[1m",  #bold
         "KRES": "\\x1B[0m",  #reset
         "KITA": "\\x1B[3m",  #italics
         "KUND": "\\x1B[4m",  #underline
         "KSTR": "\\x1B[9m",  #strikethrough
         "KINV": "\\x1B[7m",  #inverseON
         "BBLK": "\\x1B[30m", #black
         "BRED": "\\x1B[31m", #red
         "BGRN": "\\x1B[32m", #green
         "BYEL": "\\x1B[33m", #yellow
         "BBLU": "\\x1B[34m", #blue
         "BMAG": "\\x1B[35m", #magenta
         "BCYN": "\\x1B[36m", #cyan
         "BWHT": "\\x1B[37m"} #white

for c in color:
  exec(c + ' = "'  + color[c] + '"')

if __name__ == '__main__':
  print KMAG + "ansicolor" + KYEL + "." + KGRN + "py" + KNRM