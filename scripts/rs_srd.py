import sys
import slurm

try:
    observatory = sys.argv[1]
except:
    print("Specify observatory! (apo, lco, or both)")
    sys.exit()

try:
    startat = sys.argv[2]
except:
    startat = 'rs_fields'

plan = 'zeta-0'

if((observatory == 'apo') | (observatory == 'lco')):
    observatories = [observatory]
elif(observatory == 'both'):
    observatories = ['apo', 'lco']
else:
    print("No such observatory: {observatory}".format(observatory=observatory))
    sys.exit()

script_start = """
module unload python
module load miniconda/3.9
module load robostrategy/dev8
mkdir -p $ROBOSTRATEGY_DATA/allocations/{p}/targets
mkdir -p $ROBOSTRATEGY_DATA/allocations/{p}/bycarton
mkdir -p $ROBOSTRATEGY_DATA/allocations/{p}/final
cd $ROBOSTRATEGY_DATA/allocations/{p}
export MPLBACKEND=AGG
"""

cmds = ['rs_fields',
        'rs_field_rotator',
        'rs_slots',
        'rs_cadences_extract',

        'rs_targets_extract',
        'rs_priorities',
        'rs_field_targets',
        'rs_target_cadences',
        'rs_field_count',
        'rs_field_cadences',
        'rs_field_cadences_plot',
        'rs_assign',
        'rs_field_slots',
        'rs_allocate',
        'rs_allocate_plot',
        'rs_assign_prep',
        'rs_assign_final',
        'rs_assign_open',
        'rs_satisfied',
        'rs_assignments',
        'rs_completeness',
        'rs_completeness_plot',
        'rs_spares',
        'rs_spares_plot',
        'rs_cadence_html',
        'rs_html']

for observatory in observatories:

    queue = slurm.queue()
    queue.verbose = True
    queue.create(label='rs-srd-{o}-{p}'.format(o=observatory,
                                                p=plan),
                 nodes=1,
                 ppn=64,
                 walltime='60:00:00',
                 alloc='sdss-np',
                 dir='.')

    # For notchpeak use sdss-np and ppn=64
    # For kingspeak use sdss-kp and ppn=16

    script = script_start 

    start = False
    for cmd in cmds:
        if(cmd == startat):
            start = True
        if(not start):
            print("Skipping {c} ...".format(c=cmd))
        if(start):
            script = script + "S {c}\n".format(c=cmd)
            if(cmd == 'rs_fields'):
                script = script + "{c} -p {p}\n".format(p=plan, c=cmd)
            else:
                script = script + "{c} -o {o} -p {p}\n".format(o=observatory, p=plan, c=cmd)
            
    print(script)
    queue.append(script)
    queue.commit(hard=True, submit=True)
