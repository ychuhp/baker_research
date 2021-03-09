import multiprocessing
import os
import time
import subprocess



def a3m__seq(seq_fp, a3m_fp, e=.001, n=1, cpu=1):
    database = '/raid0/uniclust/uniclust30_2018_08/uniclust30_2018_08'
    os.system('hhblits -o temp.hhr -cpu '+str(cpu)+' -i '+seq_fp+' -d '+database+' -oa3m '+a3m_fp+' -n '+str(n)+ ' -e '+str(e))



loaddir = '/raid0/ychnh/seq'
savedir = '/raid0/ychnh/a3m'
N_WORKERS = None

# Build TASKS
TASKS = []
S = [ s[:-4] for s in os.listdir(loaddir) ] 
# Code for resuming from previous 
# Need code to delete partially completed files though. Probably can delete the latest N_WORKERS files
# T = [ t[:-4] for t in os.listdir(savedir)] 
# S = list(set(S) - set(T))

for s in S:
    in_fp = os.path.join(loaddir, s+'.seq')
    out_fp = os.path.join(savedir, s+'.a3m')
    TASKS.append( (in_fp, out_fp) )

try:
    pool = multiprocessing.Pool(N_WORKERS)
    r = pool.starmap(a3m__seq, TASKS)
except:
    pool.close()
    pool.terminate()
