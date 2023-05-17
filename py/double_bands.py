'''20230517 double up the bands to avoid crashing imv which only can't handle images of 2 bands '''
import os
from misc import parfor, run, read_hdr, band_names, run
bins = [x.strip()
        for x in os.popen('ls -1 *.bin').readlines()]

for b in bins:
		if len(b.split('doubled')) > 1:
				continue

		hf = b[:-4] + '.hdr'
		print(hf)
		samples, lines, bands = read_hdr(hf)

		nhf = b + '_doubled.hdr'
		run('cp ' + hf + ' ' + nhf)

		bn = [x.replace(' ', '_') for x in band_names(hf)]
		#print(bn)
		
		of = b + "_doubled.bin"
		if not os.path.exists(of):
				run('cat ' + b + ' ' + b + ' > ' + of)

		bn += [x + '_2' for x in bn]
		print(bn)

		run(' '.join(['envi_header_modify.py',
									nhf,
								  str(lines),
									str(samples),
									str(int(bands) * 2)] + bn))

		#  err('envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]')
