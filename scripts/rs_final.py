import slurm

plan = 'zeta-0'

script_start = """
module unload python
module load miniconda/3.9
module load robostrategy/1.2.0
cd $ROBOSTRATEGY_DATA/allocations/{p}
export MPLBACKEND=AGG
"""

cmds = ['rs_allocation_final -R NN',
        'rs_assignments_final']

queue = slurm.queue()
queue.verbose = True
queue.create(label='rs-final-{p}'.format(p=plan),
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
        script = script + "{c} -p {p}\n".format(p=plan, c=cmd)
            
print(script)
#queue.append(script)
#queue.commit(hard=True, submit=True)
