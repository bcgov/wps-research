import os
import sys
from misc import err

def envi_header_band_names(args): 
    #print("envi_header_band_names", args)
    # print out band names from envi header file
    # check if this is redundant with misc.py

    if len(args) < 2:
        print("python3 envi_header_band_names [envi .hdr file] # print band names within envi hdr file")
        sys.exit(1)
    lines, bandname_lines = open(args[1]).readlines(), []
    n_band_names, in_band_names = 0, False

    for i in range(0, len(lines)):
        line = lines[i].strip()

        if len(line.split("band names =")) > 1:
            in_band_names = True

        if in_band_names:
            n_band_names += 1
            if len(line.split("}")) < 2:
                w = line.split(',')
                line = ''.join(w[:-1]) + ',' + w[-1]
                lines[i] = line
            else: # on last band names line:
                lines[i] = line.replace(',', '')

        if in_band_names:
            bandname_lines.append(line) # track band names we have
        else:
            pass
            # non_bandname_lines.append(line) # record non-band-name lines,
        if in_band_names:
            if len(line.split("}")) > 1:
                in_band_names = False

    bandname_lines[0] = bandname_lines[0].split('{')[1]
    bandname_lines[-1] = bandname_lines[-1].strip('}')
    bandname_lines = [x.strip(',').strip() for x in bandname_lines]
    for b in bandname_lines:
        print(b)  # don't comment this out, some C programs like imv need it.
    return bandname_lines


def envi_update_band_names(args):
    # print("envi_update_band_names", [args])
    # transfer band names from one file to  another. Useful if you run a program that throws band name info away!
    from misc import args, sep, exists, pd, get_band_names_line_idx

    if len(args) < 3:
        err('envi_update_band_names.py [.hdr file with band names to use] ' +
            '[.hdr file with band names to overwrite]')

    if not exists(args[1]) or not exists(args[2]):
        err('please check input files:\n\t' + args[1] + '\n\t' + args[2])

    # put the band name fields in the expected places
    envi_header_cleanup([pd + 'envi_header_cleanup.py',
                         args[1]])
    envi_header_cleanup([pd + 'envi_header_cleanup.py',
                         args[2]])

    i_dat, o_dat = open(args[1]).read(),  open(args[2]).read()

    def get_band_names_lines(hdr):
        idx = get_band_names_line_idx(hdr)
        lines = open(hdr).readlines()
        return [lines[i] for i in idx], idx

    [bn1, ix1], [bn2, ix2] = get_band_names_lines(args[1]),\
                             get_band_names_lines(args[2])

    lines = o_dat.strip().split('\n')

    ix = 0
    for i in range(0, len(lines)):
        line = lines[i]  # for every line in the output file...
        if i in ix2:  # if it's supposed to be a band-names line!
            lines[i] = bn1[ix].rstrip()  # replace it with the band-names line..
            ix += 1
    open(args[2], 'wb').write('\n'.join(lines).encode()) # write the result


def envi_header_modify(args):
    # print("envi_header_modify", args)
    from misc import pd, err, exists, sep, get_band_names_line_idx
    '''update an ENVI header file:
        - band names portion
        - image dimensions or number of bands
       20230601 update to not require a new python interpreter
    (updated 20220324)'''

    if len(args) < 6:
        print("    error:", args)
        err('     envi_header_modify.py [.hdr file to modify] [nrow] [ncol] [nband] [band 1 name]... [band n name]')

    nrow, ncol, nband = args[2], args[3], args[4]
    if not exists(args[1]):
        err('please check input files:\n\t' + args[1] + '\n\t' + args[2])

    # need to run this first to make sure the band name fields are where we expect!
    if len(args) < int(nband) + 5:
        envi_header_cleanup([None,
                             args[1]])
    
    lines = open(args[1]).read().strip().split('\n')
    
    def get_band_names_lines(hdr):
        idx = get_band_names_line_idx(hdr)
        lines = open(hdr).readlines()
        return [lines[i] for i in idx], idx
    
    [bn1, ix1] = get_band_names_lines(args[1])
    
    lines_new = []
    for i in range(0, len(lines)):
        line = lines[i]  # for every line in the output file...
    
        w = [x.strip() for x in line.split('=')]
        if len(w) > 1:
            if w[0] == 'samples': line = 'samples = ' + ncol
            if w[0] == 'lines': line = 'lines = ' + nrow
            if w[0] == 'bands': line = 'bands = ' + nband
    
        if i not in ix1:  # if it's a band-names line!
            lines_new.append(line)

    # write new header file
    bn_new = args[5: 5 + int(nband)]

    # print("bn_new", bn_new)
    if len(bn_new) != int(nband):
        err('inconsistent input')

    lines_new += ['band names = {' + bn_new[0]]
    # print([bn_new[0]])

    for i in range(1, len(bn_new)):
        lines_new[-1] += ','
        #print([bn_new[i]])
        lines_new += [bn_new[i]]
    lines_new[-1] += '}'
    # print('+w', args[1])
    open(args[1], 'wb').write('\n'.join(lines_new).encode())


def envi_header_cleanup(args):
    # print("envi_header_cleanup", args)
    from misc import parfor, run, read_hdr, pd, sep, exist
    '''Clean up envi header so that they can be opened in IMV
    20230601: add option to run without a new python interpreter
    20230524:
    	envi_header_cleanup.py [input file] # if input file is .bin, will redirect to .hdr
    
    20220514:
        default:
            all hdr in present folder. Process in parallel!
    '''
    if len(args) < 2:
        # err("python3 envi_header_cleanup.py [input envi header filename .hdr]")
        lines = [x.strip() for x in os.popen('ls -1 *.hdr').readlines()]
        found = False
        jobs = []
        for line in lines:
            if exist(line):
                c = 'python3 ' + __file__ + ' ' + line
                jobs.append(c)
                found = True
        if not found:
            err("file not found: " + args[1])
    
        parfor(run, jobs, 8)
        sys.exit(0)
    
    in_file = args[1]

    # if a .bin file is provided, switch to the associated .hdr file:
    if in_file[-4:] == '.bin':
    	in_file = '.'.join(in_file.split('.')[:-1] + ['hdr'])

    
    # also record the path to the associated .bin file    
    base_file_path = '.'.join(in_file.split('.')[:-1])
    # read the lines from the header file:
    data = open(in_file).read().strip()
    n_band_names, in_band_names, nb = 0, False, 0
    data = data.replace("description = {\n", "description = {")
    data = data.replace("band names = {\n", "band names = {")
    lines, non_bandname_lines = data.split("\n"), []
    bandname_lines = []
    
    # clear the description field
    lines_new = []
    for i in range(len(lines)):
        if len(lines[i].split('description =')) < 2:
            lines_new.append(lines[i])
    lines = lines_new
    
    for i in range(len(lines)):
        line = lines[i].strip()
        w = [x.strip() for x in line.split("=")]
        if len(w) > 1:
            if w[0].strip() == 'bands':
                nb = int(w[1].strip())
            lines[i] = ' = '.join([x.strip() for x in w])
        line = lines[i].strip()
    
        if len(line.split("band names")) > 1:
            in_band_names = True
        # print(line + (" TRUE" if in_band_names else ""))
    
        if in_band_names:
            n_band_names += 1
            if len(line.split("}")) < 2:
                w = line.split(',')
                line = ''.join(w[:-1]) + ',' + w[-1]
                lines[i] = line
            else: # on last band names line:
                lines[i] = line.replace(',', '')
    
        if in_band_names:
            #print("*", line)
            bandname_lines.append(line) # track band names we have
        else:
            non_bandname_lines.append(line) # record non-band-name lines,
            # in case we need to fill the band-names in
    
        if in_band_names:
            if len(line.split("}")) > 1:
                in_band_names = False
    
    if nb != n_band_names:
        if n_band_names > nb:
            # probably should throw an error here!
            # print("n_band_names", n_band_names, "nb", nb)
            bandname_lines = bandname_lines[:nb]
            bandname_lines[-1] = bandname_lines[-1].strip() + "}"
        if n_band_names > 0 and n_band_names < nb:
            bandname_lines[-1] = bandname_lines[-1].strip().strip('}')
            for i in range(1, nb + 1):
                if i > n_band_names:
                    bandname_lines[-1] = bandname_lines[-1].strip().strip("}").strip(",") + ','
                    pre = "band names = {" if i == 1 else ""
                    bandname_lines.append(pre + "Band " + str(i) + ",")
            bandname_lines[-1] = bandname_lines[-1].strip().strip(',') + "}"
    
        if n_band_names == 0:
            bandname_lines.append("band names = {Band 1,")
            for i in range(1, nb):
                bandname_lines.append("Band " + str(i + 1) + ",")
            bandname_lines[-1] = bandname_lines[-1].strip().strip(",") + "}"
    
    print(bandname_lines)
    if [x.strip().lower() for x in bandname_lines] == ["band names = {band 1}"]:
        base_filename = base_file_path.split(os.path.sep)[-1]
        bandname_lines = ["band names = {" + base_filename.strip() + "}"]
    bandname_lines[-1] = bandname_lines[-1].replace(',', '') # no comma in last band names record
    lines = non_bandname_lines + bandname_lines
    data = ('\n'.join(lines)).strip()
    
    # print(data)
    # sys.exit(1)
    open(in_file + '.bak', 'wb').write(open(in_file).read().encode())
    open(in_file, 'wb').write(data.encode())
    
    # now trim the band names strings
    band_names = [x.strip() for x in envi_header_band_names([pd + sep + 'envi_header_band_names.py', in_file])]
    samples, lines, bands = read_hdr(in_file)
    envi_header_modify([None,
                        in_file,
                        lines,
                        samples,
                        bands] +
                       band_names + ['1'])


def envi_header_cat(args):
    # print("envi_header_cat", args)
    from misc import exists, pd
    if len(args) < 4:
        err("envi_header_cat.py [.hdr file #1] " +
            "[.hdr file #2] [output .hdr file] #" +
            "[optional prefix for bandnanes from .hdr file #1] " +
            "[optional prefix for bandnames from .hdr file #2]" +
            " #  n.b. first gets appended onto second")
    
    pre1, pre2 = '', ''
    
    if len(args) > 4:
        pre1 = args[4]
    
    if len(args) > 5:
        pre2 = args[5]
    
    def get_band_names_lines(data):
        band_name_lines, in_band_names = [], False
        lines = [x.strip() for x in data.strip().split("\n")]
        for i in range(0, len(lines)):
            if len(lines[i].split("band names =")) > 1:
                in_band_names = True
    
            if in_band_names:
                # print(lines[i])
                band_name_lines.append(lines[i])
                if len(lines[i].split("}")) > 1:
                    in_band_names = False
        return band_name_lines
    
    if not exists(args[1]) or not exists(args[2]):
        err("please check input files:\n\t" + args[1] + "\n\t" + args[2])
    
    envi_header_cleanup([pd + 'envi_header_cleanup.py',
                         args[1]])
    envi_header_cleanup([pd + 'envi_header_cleanup.py',
                         args[2]]) # should really call directly, whatever
    
    i_dat, o_dat = open(args[1]).read(), open(args[2]).read()
    bn_1, bn_2 = get_band_names_lines(i_dat), get_band_names_lines(o_dat)
    lines1 , lines2 = i_dat.strip().split('\n'), o_dat.strip().split('\n')
    
    # print(i_dat)
    # print(o_dat)
    band_count = len(bn_1) + len(bn_2) # add band counts
    # print("band_count", band_count)
    
    if lines2[-1] not in bn_2:
        print("unexpected header formatting")
    
    lines2[-1] = lines2[-1].strip().strip('}') + ','
    bn_1[0] = bn_1[0].split('{')[1]
    bn_2[-1] = bn_2[-1].strip().strip('}') + ','
    for i in range(len(bn_1)):
        bn_1[i] = pre1 + bn_1[i]
    
    for i in range(len(lines2)):
        if lines2[i] == bn_2[0]:
            lines2[i] = lines2[i].replace("band names = {", "band names = {" + pre2)
        else:
            if lines2[i] in bn_2:
                lines2[i] = pre2 + lines2[i]
    lines2 = lines2 + bn_1
    
    for i in range(len(lines2)):
        if len(lines2[i].split('bands ')) > 1:
            lines2[i] = lines2[i].split('=')[0] + '= ' + str(band_count)
    
        if len(lines2[i].split("description =")) > 1:
            lines2[i] = "description = {" + args[3][:-4] + '.bin}'
    
    f = open(args[3], "wb")
    if not f:
        err("failed to open output file: " + args[3])
    else:
        print("+w", args[3])
        f.write('\n'.join(lines2).encode())
        f.close()
