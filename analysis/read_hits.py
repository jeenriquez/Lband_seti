
import pandas as pd


filename = 'spliced_.....dat'



try:
    df = pd.read_csv(filename, sep=",|=", header=None,engine='python')
    df = pd.read_csv(filename,sep="\t",header=9,engine='python')
except:
    IOError('Error opening file: %s'%filename)



df3=df3.ix[:,:-1]
df3.columns=columns


----------------------------

df = pd.DataFrame(columns=columns)

----------------------------

def make_table(filename,only_drifting=False):
    '''
    '''

    columns = ['TopHitNum','DriftRate', 'SNR', 'Freq', 'ChanIndx', 'FreqStart', 'FreqEnd', 'CoarseChanNum', 'FullNumHitsInRange']

    file_dat = open(filename)
    hits = file_dat.readlines()

    #Info from header
    FileID = hits[1].strip().split(':')[-1].strip()
    Source = hits[3].strip().split(':')[-1].strip()

    MJD = hits[4].strip().split('\t')[0].split(':')[-1].strip()
    RA = hits[4].strip().split('\t')[1].split(':')[-1].strip()
    DEC = hits[4].strip().split('\t')[2].split(':')[-1].strip()

    #Info from individual Hits
    all_hits = [hit.strip().split('\t') for hit in hits[9:]]

    TopHitNum = zip(*all_hits)[0]
    DriftRate = zip(*all_hits)[1]
    SNR = zip(*all_hits)[2]
    Freq = zip(*all_hits)[3]
    ChanIndx = zip(*all_hits)[5]
    FreqStart = zip(*all_hits)[6]
    FreqEnd = zip(*all_hits)[7]
    CoarseChanNum = zip(*all_hits)[10]
    FullNumHitsInRange = zip(*all_hits)[11]


    data = {'TopHitNum':TopHitNum,
            'DriftRate':[float(df) for df in DriftRate],
            'SNR':[float(ss) for ss in SNR],
            'Freq':[float(ff) for ff in Freq],
            'ChanIndx':ChanIndx,
            'FreqStart':FreqStart,
            'FreqEnd':FreqEnd,
            'CoarseChanNum':CoarseChanNum,
            'FullNumHitsInRange':FullNumHitsInRange
            }

    df_data = pd.DataFrame(data)
    if only_drifting:
        df_data = df_data[df_data['DriftRate'] != '  0.000000']

    df_data = df_data.apply(pd.to_numeric)

    #Adding header information.

    df_data['FileID'] = FileID
    df_data['Source'] = Source
    df_data['MJD'] = MJD
    df_data['RA'] = RA
    df_data['DEC'] = DEC

    return df_data
