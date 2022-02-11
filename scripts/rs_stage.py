import sys
import slurm

try:
    observatory = sys.argv[1]
except:
    print("Specify observatory! (apo, lco, or both)")
    sys.exit()

try:
    stage = sys.argv[2]
except:
    print("Specify stage! (open or filler)")
    sys.exit()

plan = 'zeta-0'

if((observatory == 'apo') | (observatory == 'lco')):
    observatories = [observatory]
elif(observatory == 'both'):
    observatories = ['apo', 'lco']
else:
    print("No such observatory: {observatory}".format(observatory=observatory))
    sys.exit()

if(stage == 'open'):
    stage_option = '-O'
elif(stage == 'filler'):
    stage_option = '-F'
else:
    print("No such stage: {stage}".format(stage=stage))
    sys.exit()

script_start = """
module unload python
module load miniconda/3.9
module load robostrategy/1.2.0
cd $ROBOSTRATEGY_DATA/allocations/{p}
export MPLBACKEND=AGG
"""

cmds = ['rs_targets_extract',
        'rs_priorities',
        'rs_field_targets',
        'rs_assign_{s}'.format(s=stage),
        'rs_satisfied',
        'rs_completeness',
        'rs_completeness_plot',
        'rs_spares',
        'rs_spares_plot',
        'rs_html']

for observatory in observatories:

    queue = slurm.queue()
    queue.verbose = True
    queue.create(label='rs-{s}-{o}-{p}'.format(o=observatory,
                                               p=plan,
                                               s=stage),
                 nodes=1,
                 ppn=64,
                 walltime='60:00:00',
                 alloc='sdss-np',
                 dir='.')

    # For notchpeak use sdss-np and ppn=64
    # For kingspeak use sdss-kp and ppn=16

    script = script_start 

    start = False
    startat = cmds[0]
    for cmd in cmds:
        if(cmd == startat):
            start = True
        if(not start):
            print("Skipping {c} ...".format(c=cmd))
        if(start):
            script = script + "S {c}\n".format(c=cmd)
            if(cmd == 'rs_assign_{s}'.format(s=stage)):
                script = script + "{c} -o {o} -p {p}\n".format(o=observatory, p=plan, c=cmd)
            else:
                script = script + "{c} {s} -o {o} -p {p}\n".format(s=stage_option, o=observatory, p=plan, c=cmd)
            
    print(script)
    #queue.append(script)
    #queue.commit(hard=True, submit=True)
