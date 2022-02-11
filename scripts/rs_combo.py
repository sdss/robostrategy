import slurm
import sys

try:
    stage = sys.argv[1]
except:
    print("Must specify stage! (srd, open, or filler)")
    sys.exit()

if(stage == 'srd'):
    stage_option = ''
elif(stage == 'open'):
    stage_option = '-O'
elif(stage == 'filler'):
    stage_option = '-F'
else:
    print("No such stage: {stage}".format(stage=stage))
    sys.exit()

observatory = 'both'
plan = 'zeta-0'

queue = slurm.queue()
queue.verbose = True
queue.create(label='rs-{stage}-combo-{o}-{p}'.format(o=observatory,
                                                     p=plan,
                                                     stage=stage),
             nodes=1,
             ppn=64,
             walltime='60:00:00',
             alloc='sdss-np',
             dir='.')


# For notchpeak use sdss-np and ppn=64
# For kingspeak use sdss-kp and ppn=16

script_start = """
module unload python
module load miniconda/3.9
module load robostrategy/1.2.0
cd $ROBOSTRATEGY_DATA/allocations/{p}
export MPLBACKEND=AGG
"""

script = script_start 
script = script + "cp $RSCONFIG_DIR/etc/robostrategy-{p}.cfg $ROBOSTRATEGY_DATA/allocations/{p}\n".format(p=plan)

cmds = ['rs_completeness',
        'rs_completeness_plot',
        'rs_html',
        'rs_stats']

start = False
startat = cmds[0]
for cmd in cmds:
    if(cmd == startat):
        start = True
    if(not start):
        print("Skipping {c} ...".format(c=cmd))
    if(start):
        script = script + "S {c}\n".format(c=cmd)
        script = script + "{c} {s} -o {o} -p {p}\n".format(c=cmd, p=plan,
                                                           o=observatory,
                                                           s=stage_option)
            
print(script)
#queue.append(script)
#queue.commit(hard=True, submit=True)
