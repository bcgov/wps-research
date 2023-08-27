'''copy map info from one ENVI header file to another'''
import os
import sys
from envi import envi_header_cleanup
from misc import sep, exists, pd, err, run, get_map_info_lines_idx, args

def envi_header_copy_mapinfo(args):
    if len(args) < 3:
        err('envi_header_copy_mapinfo.py [source .hdr file] [dest .hdr file to modify]')
    if not exists(args[1]) or not exists(args[2]): 
        err('please check input files:\n\t' + args[1] + '\n\t' + args[2])

    # need to run this first to make sure the band name fields are where we expect!
    # run('python3 ' + pd + 'envi_header_cleanup.py ' + args[1])
    # run('python3 ' + pd + 'envi_header_cleanup.py ' + args[2])
    print("args", args)
    envi_header_cleanup(['envi_header_cleanup', args[1]])
    envi_header_cleanup(['envi_header_cleanup', args[2]])    
    lines = open(args[1]).read().strip().split('\n')
    lines2 = open(args[2]).read().strip().split('\n')

    X = get_map_info_lines_idx(args[1])
    Y = get_map_info_lines_idx(args[2])

    for i in [0, 1]:
        if X[i] is not None:
            m = lines[X[i]]
            if Y[i] is not None:
                lines2[Y[i]] = m
            else:
                lines2.append(m)
    print('+w', args[2])
    open(args[2], 'wb').write('\n'.join(lines2).encode())


if __name__ == '__main__':
    envi_header_copy_mapinfo(args)
