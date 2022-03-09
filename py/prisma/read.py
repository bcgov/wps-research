'''Script for reading ASI PRISMA L2D data (.he5) to ENVI .bin file.
Tested on 20210729 date. VNIR best match: r,g,b=37,49,58
        
    * write ENVI headers with map info'''
import os
import sys
import h5py
import math
import numpy as np
sep = os.path.sep
args = sys.argv

def err(m):
    print("Error: " + str(m)); sys.exit(1)

if len(args) < 2:
    err("usage:\n\tprisma/read.py [input PRISMA hdf5 file]")

spec = """Band_Index,WL_VNIR[nm],FWHM_VNIR[nm],WL_SWIR[nm],FWHM_SWIR[nm]
1,1004.965454,12.62831306,2497.115479,9.728999138
2,994.9711914,12.93788719,2490.219238,9.218007088
3,984.3811646,12.97115517,2483.792969,9.122716904
4,973.9871216,12.91279602,2477.054932,9.811144829
5,963.6483765,12.94441032,2469.627197,9.328546524
6,952.8931885,13.46378422,2463.030273,9.059476852
7,941.2490234,13.27760506,2456.585693,9.568544388
8,930.7337646,12.75405407,2449.142334,9.607460022
9,920.5718994,12.88076591,2442.403076,8.924209595
10,910.0667725,13.06629372,2435.544189,9.609920502
11,899.4233398,13.11040783,2428.667725,9.633436203
12,888.7026367,13.1060648,2421.237305,9.907518387
13,878.0777588,13.0431881,2414.356689,9.114235878
14,867.3642578,13.17148495,2407.604492,9.83366394
15,856.5948486,13.14589977,2400.035889,9.842931747
16,845.8317261,13.08838367,2393.038818,9.4485569
17,835.1489258,13.00885773,2386.061768,9.870838165
18,824.5411987,12.98768139,2378.770996,9.828164101
19,813.9327393,12.95625687,2371.552246,9.707830429
20,803.3327637,12.94817162,2364.594482,9.642411232
21,792.7487183,12.85449696,2357.293701,10.13381481
22,782.28125,12.7641325,2349.791504,9.74174881
23,771.8952026,12.74426174,2342.822754,9.733902931
24,761.4781494,12.80502892,2335.526367,10.23498631
25,751.0913086,12.62770176,2327.824219,9.873425484
26,740.7712402,12.56452942,2320.895508,9.861345291
27,730.5769043,12.41900063,2313.200684,10.38665962
28,720.5016479,12.3491993,2305.722656,9.817890167
29,710.3485718,12.48248196,2298.609375,10.14062691
30,700.3587036,11.98217583,2290.82666,10.22005653
31,690.750061,12.03603745,2283.493408,9.950322151
32,680.7589722,11.97917271,2276.053711,10.39184952
33,671.0860596,11.83851051,2268.28833,10.10565662
34,661.5158081,11.71825027,2260.866455,10.39723682
35,652.0404663,11.63319969,2253.110352,10.35341549
36,642.5855103,11.54829311,2245.448486,10.24898434
37,633.2979736,11.14274788,2237.904053,10.47025681
38,624.4001465,11.21997261,2230.007568,10.48997116
39,615.3416748,11.07150745,2222.42627,10.31995964
40,606.579895,10.92410946,2214.625,10.56896687
41,597.6361084,10.82665634,2206.843018,10.25385094
42,588.9606934,10.55413246,2199.135254,10.70757294
43,580.4658203,10.39586353,2191.100342,10.53475571
44,572.1085205,10.31268024,2183.420166,10.42951012
45,563.8278198,10.17391586,2175.344238,10.81814098
46,555.6419678,10.05979729,2167.484863,10.4052906
47,547.5363159,10.01752567,2159.563965,10.97972584
48,539.5068359,9.795125961,2151.38623,10.72788334
49,531.6742554,9.669556618,2143.465576,10.71089363
50,523.927002,9.641452789,2135.510254,10.90403748
51,516.1654663,9.507074356,2127.337158,10.83949375
52,508.6680603,9.366909027,2119.231445,10.96091175
53,501.1333923,9.292699814,2111.039063,10.97451687
54,493.6803589,9.206349373,2102.821289,11.00893593
55,486.4165039,9.007447243,2094.625244,11.01621532
56,479.1698303,8.908111572,2086.382324,11.20486259
57,471.9405212,8.914631844,2077.991455,11.09249592
58,464.7201233,8.920166969,2069.795654,11.2635088
59,457.3534241,9.059168816,2061.378662,11.09842205
60,450.0111389,9.123975754,2053.007813,11.17084217
61,442.6363831,9.195631027,2044.680908,11.12836361
62,435.2785645,9.226050377,2036.260742,11.42365456
63,427.9563599,9.332583427,2027.726685,11.25815964
64,420.4064331,9.750845909,2019.321411,11.43357944
65,412.4606628,10.37718678,2010.661377,11.38755894
66,403.6150818,11.35224819,2002.110596,11.53911209
67,,,1993.548218,11.40911007
68,,,1984.853027,11.81587887
69,,,1976.012939,11.57380199
70,,,1967.341797,11.73533535
71,,,1958.62439,11.54941559
72,,,1949.900757,11.72707653
73,,,1941.110718,11.95355701
74,,,1932.26001,11.81017399
75,,,1923.385742,12.18462849
76,,,1914.301392,11.8970871
77,,,1904.934692,12.72552586
78,,,1896.091309,11.63577652
79,,,1887.080933,11.77403545
80,,,1878.742554,12.56736851
81,,,1868.17334,11.46572495
82,,,1859.558716,12.36181641
83,,,1850.554321,12.62172794
84,,,1841.325562,12.58400536
85,,,1832.027222,12.48864269
86,,,1822.441284,13.07937717
87,,,1813.05127,12.37173557
88,,,1803.59021,12.54814148
89,,,1793.953125,12.72824287
90,,,1784.717285,12.59507179
91,,,1775.117798,12.47487736
92,,,1765.512573,12.71340752
93,,,1755.833008,13.0777874
94,,,1746.219238,12.97546673
95,,,1736.488403,12.83528137
96,,,1726.651611,12.81098557
97,,,1716.858887,13.02038288
98,,,1707.094482,13.2541399
99,,,1697.294312,13.22678375
100,,,1687.42688,13.04662704
101,,,1677.319336,12.99486542
102,,,1667.185181,13.46423435
103,,,1656.932983,12.85817623
104,,,1647.231567,13.39513588
105,,,1637.091919,13.57167625
106,,,1627.020996,13.66740322
107,,,1616.833618,14.02093983
108,,,1606.491333,13.76013184
109,,,1596.245361,13.90128708
110,,,1585.859863,13.92940044
111,,,1575.627319,13.81271458
112,,,1565.368774,13.99826717
113,,,1554.816772,14.14927197
114,,,1544.226196,13.93795967
115,,,1533.776367,14.10292816
116,,,1523.22229,14.08894348
117,,,1512.633301,14.15331078
118,,,1502.023438,14.32816219
119,,,1491.429199,14.19109821
120,,,1480.842163,14.35108376
121,,,1469.930786,14.44627094
122,,,1459.315674,13.88232517
123,,,1449.188843,13.9790535
124,,,1438.465942,14.87912655
125,,,1427.374634,14.24291229
126,,,1416.537354,14.51218128
127,,,1405.626953,14.59148216
128,,,1394.754028,14.52481651
129,,,1383.279785,15.15550327
130,,,1372.911743,14.68958855
131,,,1361.053101,15.28233051
132,,,1349.78772,14.76456547
133,,,1339.129395,14.64970589
134,,,1328.299316,14.81002998
135,,,1317.256592,14.73632336
136,,,1306.218018,14.65475368
137,,,1295.421875,14.6244278
138,,,1284.487793,14.7072401
139,,,1273.496338,14.74774456
140,,,1262.532227,15.11242199
141,,,1250.979858,14.98299408
142,,,1240.214478,14.5253334
143,,,1229.185181,14.88925838
144,,,1217.863525,14.60076618
145,,,1207.273682,14.74438667
146,,,1196.339355,14.46939373
147,,,1185.588379,14.56452274
148,,,1174.714233,14.45989418
149,,,1163.676147,14.86172199
150,,,1152.650146,14.47423077
151,,,1142.070313,14.33642483
152,,,1131.30481,14.54416561
153,,,1120.675903,14.18616867
154,,,1109.889404,14.60808372
155,,,1099.277588,14.23894405
156,,,1088.760986,14.28531265
157,,,1078.216064,14.16227055
158,,,1067.7948,14.0451746
159,,,1057.57373,13.82085896
160,,,1047.675049,13.63140965
161,,,1037.987793,13.56242466
162,,,1029.343994,12.9564867
163,,,1018.535706,12.97453213
164,,,1008.644287,12.45164967
165,,,998.9082031,12.67055607
166,,,988.9179077,12.28469372
167,,,979.223999,12.13259125
168,,,969.8449097,12.15050888
169,,,959.973938,12.20068169
170,,,951.4014282,11.01296139
171,,,943.3579102,10.94130802
172,,,934.6009521,11.1797266
173,,,925.6868286,11.056283"""

def read_csv(f):
    lines = [x.strip().split(',') for x in f.strip().split('\n')]
    hdr = lines[0]
    lines, dat = lines[1:], {k: [] for k in hdr}
    for line in lines:
        for i in range(0, len(line)):
            dat[hdr[i]].append(line[i])
    #for k in dat: print(k, dat[k])
    return dat
spec = read_csv(spec) # print(spec.keys())

def write_hdr(hfn, samples, lines, bands, dsn=None, other=''):
    # WL_VNIR[nm] WL_SWIR[nm] # SWIR_Cube VNIR_Cube
    bands = 1 if bands is None else bands
    rgb_d = None # rgb band indexes from 1
    SWIR = (dsn == 'SWIR_Cube')
    VNIR = (dsn == 'VNIR_Cube')
    rgb_i = None
    if VNIR:  # 630 nm red, 532 nm green, 465 nm blue
        rgb_t, rgb_i, rgb_d = [630, 532, 465], [0,0,0], [None, None, None]
        w_l = spec['WL_VNIR[nm]']
        for i in range(len(w_l)):
            if w_l[i] != '':
                w = int(round(float(w_l[i])))
                di = [abs(rgb_t[j] - w) for j in range(3)]
                for j in range(3):
                    if rgb_d[j] is None or di[j] < rgb_d[j]:
                        rgb_d[j] = di[j]
                        rgb_i[j] = i
                    else:
                        pass
        rgb_i = [x + 1 for x in rgb_i] # 1 index
    w_len = ': ' if (SWIR or VNIR) else ''
    if (SWIR or VNIR):
        w_len += str(int(round(float(spec['WL_SWIR[nm]'][0] if SWIR
                     else spec['WL_VNIR[nm]'][0])))) + 'nm'
    print('+w', hfn)
    rgb_s = (','.join([str(x) for x in rgb_i])) if rgb_i else ''
    lines = ['ENVI' + ('\ndescription = {rgb=' + rgb_s + '}' if VNIR else ''),
             'samples = ' + str(samples),
             'lines = ' + str(lines),
             'bands = ' + str(bands),
             'header offset = 0',
             'file type = ENVI Standard',
             'data type = 4',
             'interleave = bsq',
             'byte order = 0']
    if other != '':
        lines += other.split('\n')
    lines += ['band names = {Band 1' + w_len]
    if bands > 1:
        for i in range(1, bands):
            lines[-1] += ','
            w_len = ': ' if (SWIR or VNIR) else ''
            if(SWIR or VNIR):
                w_len += str(int(round(float(spec['WL_SWIR[nm]'][i] if SWIR else spec['WL_VNIR[nm]'][i])))) + 'nm'
            lines.append('Band ' + str(i + 1) + w_len)
    lines[-1] += '}'
    open(hfn, 'wb').write('\n'.join(lines).encode())

filename = sys.argv[1]
if filename[-3:] != 'he5': err("unexpected filename: .h35 expected")
fn_base = filename[:-4] # print(fn_base) 
datasets, data_sets = [], {}

def iterate(x, s="", parent=None, match=" dataset ", printed=True):
    if printed:
        print(s,x, "attrs:" + str(x.attrs.keys())) # uncomment this line for debug
    keys = None
    try:
        keys = x.keys()
        for k in keys:
            iterate(x[k], s + "**", parent=x, match=match, printed=printed)
    except:
        if len(str(x).split(match)) > 1:
            datasets.append([x, parent])
            dsn = str(x).strip().split(':')[0].split('"')[1].strip('"')
            data_sets[dsn] = [x, parent]
    #if keys is None:
    #   print(s, x)

with h5py.File(filename, "r") as f:
    for k in f.attrs.keys():
        x = f.attrs.get(k)
        print('\t' + str(k) + '=' + str(x), str(type(x)))
    
    def short(x):
        n = 1 # 2
        w = [y[:1] for y in x.split('_')]
        return '_'.join(w)

    map_info = {short(x):f.attrs.get(x)
                for x in ['Projection_Name',
                          'Projection_Id',
                          'Epsg_Code',
                          'Reference_Ellipsoid',
                          'Product_ULcorner_easting',
                          'Product_ULcorner_northing',
                          'Product_LRcorner_easting',
                          'Product_LRcorner_northing']}
    wkt = str(map_info['E_C'])
    wkt = os.popen('gdalsrsinfo -o wkt_esri epsg:' + wkt).read().strip().split()
    wkt = ''.join(wkt)
    wkt = 'coordinate system string = {' + wkt + '}'
    N_S = wkt.split(',')[0].split('Zone')[-1].strip('"')[-1]
    X_size, Y_size = None, None
    iterate(f, printed=False)  # print("fields available:", list(data_sets.keys()))
    want = ['SWIR_Cube', 'VNIR_Cube'] # the meat
    
    if False:  # enable this for more stuff.. examples of avail. stuff
        want += ['Latitude', 'Longitude',
                 'Wgs84_pos_x', 'Wgs84_pos_y', 'Wgs84_pos_z', 
                 'Cw_Swir_Matrix', 'Cw_Vnir_Matrix',
                 'Fwhm_Swir_Matrix', 'Fwhm_Vnir_Matrix']
    # print("fields selected:", str(want))
    for w in want:
        if w not in data_sets:
            err("key not found: " + str(w))

    for dsn in want:  # data set name
        w = data_sets[dsn]
        dsp = str(w[1]).strip().split('"')[1] + '/' + dsn
        dsps = str(dsp)
        dsp = dsp.strip().strip('/').split('/')
        x = f
        for i in dsp:
            x = x[i] # f['HDFEOS']['SWATHS']['PRS_L1_HCO']['Data Fields']['SWIR_Cube'][()]
        x = x[()]
        
        data = np.array(x)
        N = len(data.shape) # how many dimensions? 3 is cube. 2 is 1-band..
        nrow, ncol, nband = None, None, None # image dimensions
        fn = fn_base + '_' + (dsn.replace(' ', '_')) + '.bin'
        hn = fn[:-4] + '.hdr' # print(dsps, '->', fn)
        o_f = open(fn, 'wb')
        dt = '>f4' # default data type to write! always float32, byte order 0
        if N == 3:
            nband = data.shape[1]
            nrow, ncol = data[:,0,:].shape
            for i in range(nband):
                if i % 25 == 0:
                    print("\tband", str(i + 1), 'of', nband)
                data[:,i,:].astype(np.float32).tofile(o_f, '', dt)
        elif N == 2:
            nrow, ncol = data.shape
            data.astype(np.float32).tofile(o_f, '', dt)
        else:
            err('unexpected dimensions')
        if N == 3:
            pass

        X_size = abs((map_info['P_U_e'] - map_info['P_L_e']) / float(ncol))
        Y_size = abs((map_info['P_U_n'] - map_info['P_L_n']) / float(nrow))
        print("X_size", X_size, "Y_size", Y_size)
        ''' map info. Lists geographic information in the following order:
            * Projection name
            * Reference (tie point) pixel x location (in file coordinates)
            * Reference (tie point) pixel y location (in file coordinates)
            * Pixel easting
            * Pixel northing
            * x pixel size
            * y pixel size
            * Projection zone (UTM only)
            * North or South (UTM only)
            * Datum
            * Units
        '''
        m_i = [map_info['P_N'].decode('utf-8'),
               str(1),
               str(1),
               str(map_info['P_U_e']),
               str(map_info['P_U_n']),
               str(X_size),
               str(Y_size),
               map_info['P_I'].decode('utf-8'),
               'North' if N_S == 'N' else 'South',
               map_info['R_E'].decode('utf-8'),
               'units=Meters']
        mapinfo = 'map info = {' + ','.join(m_i) + '}'
        other = mapinfo + '\n' + wkt
        write_hdr(hn, ncol, nrow, nband, dsn, other=other)
        o_f.close()
