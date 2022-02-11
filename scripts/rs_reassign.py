import slurm

plan = 'zeta-0'

queue = slurm.queue()
queue.verbose = True
queue.create(label='rs-reassign-{p}'.format(p=plan),
             nodes=1,
             ppn=64,
             walltime='50:00:00',
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
script = script + "echo S rs_assign_spares apo\n"
script = script + "rs_assign_spares -o apo -p {p}\n"

script = script + "echo S rs_assign_spares lco\n"
script = script + "rs_assign_spares -o lco -p {p}\n"

script = script + "echo S rs_extra\n"
script = script + "rs_extra -p {p}\n"

queue.append(script.format(p=plan))
queue.commit(hard=True, submit=True)
