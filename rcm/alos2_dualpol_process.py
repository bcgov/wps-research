'''20230123 process JAXA data retrieved from EODMS
*** Assume each folder in present directory, is a dataset'''

FILTER_SIZE = 3
import os
import sys
sep = os.path.sep
my_path = sep.join(os.path.abspath(__file__).split(sep)[:-1]) + sep
sys.path.append(my_path + ".." + sep + 'py')
from misc import pd, sep, exist, args, cd, err
QUAD_POL = False # mode flag: base case is dual-pol ( revise for Quad-pol ) 

def run(x):
    cmd = ' '.join(x)
    print(cmd)
    a = os.system(cmd)

snap, ci = '/usr/local/snap/bin/gpt', 0 # assume we installed snap
if not exist(snap):
    snap = '/opt/snap/bin/gpt'  # try another location if that failed
if not exist(snap):
    snap = '/home/' + os.popen('whoami').read().strip() + sep + 'snap' + sep + 'bin' + sep + 'gpt'

print(snap)

dirs = [f for f in os.listdir() if os.path.isdir(f)]
# print(dirs)


i = 0
for d in dirs:
    print("i=", str(i + 1), "of", str(len(dirs)))
    print(d)

    # look for VOL file
    vol_files = [f for f in os.listdir(d) if len(f.split('VOL')) > 1]
    if len(vol_files) > 1:
        err('expected only one *VOL* file')

    p_0 = d + sep + vol_files[0] # 'manifest.safe'  # input
    p_1 = d + sep + '01_Mlk.dim'
    p_2 = d + sep + '02_Cal.dim' # calibrated product
    p_3 = d + sep + '03_Mtx.dim'
    p_4 = d + sep + '04_Rtf.dim'
    p_5 = d + sep + '05_Box.dim'
    p_6 = d + sep + '06_Ter.dim'
    p_7 = d + sep + '07_Box.dim'
    print(p_0)
    
    if not exist(p_1):
        run([snap,
             'Multilook',
             '-PnAzLooks=2',
             '-PnRgLooks=4',
             '-Ssource=' + p_0,
             '-t ' + p_1])
    print(p_1)

'''
/opt/snap/bin/gpt Calibration -h 
Usage:
  gpt Calibration [options] 

Description:
  Calibration of products


Source Options:
  -Ssource=<file>    Sets source 'source' to <filepath>.
                     This is a mandatory source.

Parameter Options:
  -PauxFile=<string>                                    The auxiliary file
                                                        Value must be one of 'Latest Auxiliary File', 'Product Auxiliary File', 'External Auxiliary File'.
                                                        Default value is 'Latest Auxiliary File'.
  -PcreateBetaBand=<boolean>                            Create beta0 virtual band
                                                        Default value is 'false'.
  -PcreateGammaBand=<boolean>                           Create gamma0 virtual band
                                                        Default value is 'false'.
  -PexternalAuxFile=<file>                              The antenna elevation pattern gain auxiliary data file.
  -PoutputBetaBand=<boolean>                            Output beta0 band
                                                        Default value is 'false'.
  -PoutputGammaBand=<boolean>                           Output gamma0 band
                                                        Default value is 'false'.
  -PoutputImageInComplex=<boolean>                      Output image in complex
                                                        Default value is 'false'.
  -PoutputImageScaleInDb=<boolean>                      Output image scale
                                                        Default value is 'false'.
  -PoutputSigmaBand=<boolean>                           Output sigma0 band
                                                        Default value is 'true'.
  -PselectedPolarisations=<string,string,string,...>    The list of polarisations
  -PsourceBands=<string,string,string,...>              The list of source bands.

Graph XML Format:
  <graph id="someGraphId">
    <version>1.0</version>
    <node id="someNodeId">
      <operator>Calibration</operator>
      <sources>
        <source>${source}</source>
      </sources>
      <parameters>
        <sourceBands>string,string,string,...</sourceBands>
        <auxFile>string</auxFile>
        <externalAuxFile>file</externalAuxFile>
        <outputImageInComplex>boolean</outputImageInComplex>
        <outputImageScaleInDb>boolean</outputImageScaleInDb>
        <createGammaBand>boolean</createGammaBand>
        <createBetaBand>boolean</createBetaBand>
        <selectedPolarisations>string,string,string,...</selectedPolarisations>
        <outputSigmaBand>boolean</outputSigmaBand>
        <outputGammaBand>boolean</outputGammaBand>
        <outputBetaBand>boolean</outputBetaBand>
      </parameters>
    </node>
  </graph>
'''

    if not exist(p_2):
        run([snap, 'Calibration',
             '-Ssource=' + p_1,
             '-t ' + p_2,
             '-PoutputImageInComplex=true'])
    print(p_2)

    # check number of channels to determine if dual-pol or quad-pol dataset
    n_channels = os.popen('grep "Number of SAR channels" ' + p_2).read().strip().split('\n')[0]
    # print(n_channels) 
    n_channels = int(n_channels.split('>')[1].split('<')[0])
    # print(n_channels)
    if(n_channels == 4):
        print("DUAL-POL MODE")
    elif(n_channels == 8):
        print("QUAD-POL MODE")
        QUAD_POL = True
    else:
        print("UNEXPECTED NUMBER OF CHANNELS")
        sys.exit(1)

    if not exist(p_3):
        run([snap, 'Polarimetric-Matrices',
             '-Ssource=' + p_2,
             '-t ' + p_3,
             '-Pmatrix=' + ('C2' if not QUAD_POL else 'T4')])
    print(p_3)

    '''
     /opt/snap/bin/gpt  Terrain-Flattening  -h 
  gpt Terrain-Flattening [options] 
Description:
  Terrain Flattening
Source Options:
  -Ssource=<file>    Sets source 'source' to <filepath>.
                     This is a mandatory source.
Parameter Options:
  -PadditionalOverlap=<double>                The additional overlap percentage
                                              Valid interval is [0, 1].
                                              Default value is '0.1'.
  -PdemName=<string>                          The digital elevation model.
                                              Default value is 'SRTM 1Sec HGT'.
  -PdemResamplingMethod=<string>              Sets parameter 'demResamplingMethod' to <string>.
                                              Default value is 'BILINEAR_INTERPOLATION'.
  -PexternalDEMApplyEGM=<boolean>             Sets parameter 'externalDEMApplyEGM' to <boolean>.
                                              Default value is 'false'.
  -PexternalDEMFile=<file>                    Sets parameter 'externalDEMFile' to <file>.
  -PexternalDEMNoDataValue=<double>           Sets parameter 'externalDEMNoDataValue' to <double>.
                                              Default value is '0'.
  -PnodataValueAtSea=<boolean>                Mask the sea with no data value (faster)
                                              Default value is 'true'.
  -PoutputSigma0=<boolean>                    Sets parameter 'outputSigma0' to <boolean>.
                                              Default value is 'false'.
  -PoutputSimulatedImage=<boolean>            Sets parameter 'outputSimulatedImage' to <boolean>.
                                              Default value is 'false'.
  -PoversamplingMultiple=<double>             The oversampling factor
                                              Valid interval is [1, 4].
                                              Default value is '1.0'.
  -PsourceBands=<string,string,string,...>    The list of source bands.
Graph XML Format:
  <graph id="someGraphId">
    <version>1.0</version>
    <node id="someNodeId">
      <operator>Terrain-Flattening</operator>
      <sources>
        <source>${source}</source>
      </sources>
      <parameters>
        <sourceBands>string,string,string,...</sourceBands>
        <demName>string</demName>
        <demResamplingMethod>string</demResamplingMethod>
        <externalDEMFile>file</externalDEMFile>
        <externalDEMNoDataValue>double</externalDEMNoDataValue>
        <externalDEMApplyEGM>boolean</externalDEMApplyEGM>
        <outputSimulatedImage>boolean</outputSimulatedImage>
        <outputSigma0>boolean</outputSigma0>
        <nodataValueAtSea>boolean</nodataValueAtSea>
        <additionalOverlap>double</additionalOverlap>
        <oversamplingMultiple>double</oversamplingMultiple>
      </parameters>
    </node>
  </graph>
    '''
    if not exist(p_4):
        run([snap, 'Terrain-Flattening',
            '-PdemName="Copernicus 30m Global DEM"',  #SRTM 1Sec HGT"',
            '-Ssource=' + p_3,
            '-t ' + p_4]) # output
    print(p_4)

    if not exist(p_5):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_4,
            '-t ' + p_5]) # output
    print(p_5)
 
    if not exist(p_6):
        run([snap, 'Terrain-Correction',
            '-PnodataValueAtSea=true',
            '-Ssource=' + p_5,
            # '-PpixelSpacingInMeter=10.0',
            ' -PdemName="Copernicus 30m Global DEM"',
            '-t ' + p_6])  # output
    print(p_6)
    
    if not exist(p_7):
        run([snap, 'Polarimetric-Speckle-Filter',
            '-Pfilter="Box Car Filter"',
            '-PfilterSize=' + str(FILTER_SIZE),
            '-Ssource=' + p_6,
            '-t ' + p_7]) # output
    print(p_7)
    

    # sys.exit(1)  # comment out to run on first set only

    i += 1


    '''
        Now need convert to ENVI, zero to NAN, and mutual coregistration 
    '''
