'''20220412 take video of PNG screenshots, present dir..
..resize to 50% and create interlaced gif

Resizing was motivated by memory limits in MS Teams
for animation display

imagemagick required:
    convert'''
import os
from misc import run
ofs = []
running = False
for f in [x.strip() for x in os.popen('ls -1 Screenshot*').readlines()]:
    of = f.replace('Screenshot from ', '').replace(' ', '_')
    if not os.path.exists(of):
        run('convert "' + f + '" -resize 50% ' + of + ' &')
        running = True
    ofs.append(of)
    
cmd = 'convert -delay 33 ' + ' '.join(ofs) + ' antonov.gif'
if running == False: # ready to make gif
    run(cmd)
else:
    print("Run again to generate interlaced gif")


