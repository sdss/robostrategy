# @Author: Michael R. Blanton
# @Date: Aug 3, 2018
# @Filename: allocate.py
# @License: BSD 3-Clause
# @Copyright: Michael R. Blanton

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import numpy as np
import fitsio
import collections
import matplotlib.pyplot as plt
import robostrategy
import robostrategy.header
import robostrategy.slots
import roboscheduler.cadence as rcadence
import roboscheduler.scheduler as scheduler
from ortools.linear_solver import pywraplp

try:
    import mpl_toolkits.basemap as basemap
except ImportError:
    basemap = None




def option_epochs_done(option_cadence=None, current_cadence=None,
                       current_exposures_done=None,
                       skybrightness_only=False):
    """Assess field options given status

    Parameters
    ----------

    option_cadence : str
        cadence option to check

    current_cadence : str
        current cadence for this field

    current_exposures_done : ndarray of np.int32
        current exposures that are done

    skybrightness_only : bool
        whether the full sky conditions should be checked (False) or not (True)

    Returns
    -------

    ok: bool
        whether the option's cadence can be used

    epochs_done : ndarray of np.int32
        which epochs in option_cadence should be considered done
         
    Notes
    -----

    This associates current exposures done with the earliest of option's
    exposures, avoiding if possible associating a current bright time exposure
    with a dark time exposure in the option's cadence.
"""
    clist = rcadence.CadenceList()
    if(len(clist.cadences) == 0):
        raise ValueError("CadenceList must be initialized")

    current_epochs = clist.cadences[current_cadence].epochs[current_exposures_done]

    if(len(current_epochs) == 0):
        return(False, np.array([], dtype=np.int32))

    current_epochs, nexp_epochs = np.unique(current_epochs, return_counts=True)
    if(np.any(nexp_epochs < clist.cadences[current_cadence].nexp[current_epochs])):
        print("Incomplete epoch in list of done exposures", flush=True)
        return(False, np.array([], dtype=np.int32))

    print(" ... Calling specific_cadence_consistency()", flush=True)
    ok, epochs_list = clist.specific_cadence_consistency(current_cadence, option_cadence,
                                                         current_epochs, limit=1000)
    if(not ok):
        return(False, np.array([], dtype=np.int32))

    # Now pick an option which minimizes the number of originally-bright cadences
    # associated with dark epochs
    nbright_in_dark = np.zeros(len(epochs_list), dtype=np.int32)
    lowest = -1
    print(" ... Cycling through epochs_list", flush=True)
    for indx, epochs in enumerate(epochs_list):
        nbright_in_dark[indx] = (clist.cadences[current_cadence].skybrightness[current_epochs] >
                                 clist.cadences[option_cadence].skybrightness[epochs]).sum()
        if(nbright_in_dark[indx] == 0):
            lowest = indx
            break
    if(lowest < 0):
        lowest = np.argmin(nbright_in_dark)

    return(True, np.array(epochs_list[lowest], dtype=np.int32))


class AllocateLST(object):
    """LST allocation object for robostrategy

    Parameters
    ----------

    slots : robostrategy.slots.Slots object
        information about available time in survey

    field_slots : ndarray
        for each field-cadence combination, which slots are available

    field_options : ndarray
        for each field-cadence combination, what cadence options to choose from

    seed : int, np.int32
        random seed for solution selection (default 100)

    dark_prefer : float, np.float32
        factor by which to prefer a dark cadence (multiplies value)

    Attributes
    ----------

    slots : robostrategy.slots.Slots object
        information about available time in survey

    field_slots : ndarray
        for each field-cadence combination, which slots are available

    field_options : ndarray
        for each field-cadence combination, what cadence options to choose from

    nfields : int
        number of fields

    seed : int, np.int32
        random seed for solution selection

    allocinfo : OrderedDict
        complicated object storing linear programming variables
        (set by construct() method)

    field_array : ndarray
        for each field, solution as to how to observe it, with fields:

         * 'fieldid' : field identifier
         * 'cadence' : cadence name
         * 'nallocated' : observations allocated in this run
         * 'nallocated_sb' : same, per skybrightness
         * 'nallocated_full' : observations including already done
         * 'nallocated_full_sb' : same, per skybrightness
         * 'slots_exposures' : number of exposures allocated for each slot
         * 'slots_time' : time allocated for each slots (hours)

    Notes
    -----

    The output in slots_time is the number of exposures, times the duration
    of a nominal exposure, times the "xfactor" associated with its airmass.

    This class applies a linear programming approach to allocating
    field observations.

    The inputs are:

      * the available time as a function of LST and sky brightness (slots)
      * the slots that each field could be observed in (field_slots)
      * the cadences available for each field (field_options)

    The code constructs a linear programming problem of the following
    form. We define a set of variables indexed w_ijk, where i indexes
    the fields, j indexes the cadences available for each field, and k
    indexes the usable slots for that field and cadence (which depend
    on LST and sky brightness). There are not necessarily the same number of
    indices for each field.

    We then impose the following constraints::

      \\sum_{ij} w_{ijk} N_{ik} < T_k, 

      0 <= w_{ijk} <= A_{ij}

      \\sum_k w_{ijk} <= A_{ij}

      \\sum_{jk} (w_{ijk} / A_{ij}) <= 1,

    with A_{ij} the total time necessary for the field-cadence i-j,
    with T_k the maximum # exposures in each slot,
    and with N_{ik} expressing the time taken relative to a nominal 
    exposure of observing a particular field in a particular slot.

    Then we define a value::

      V = \\sum_{ijk} (w_{ijk} / A_{ij}) V_{ij}

    where V_{ij} is the value of getting all of a particular field-cadence.

    Then or-tools is used to solve this linear programming problem.

    Now, this linear programming programming is not exactly the problem
    we want to solve. In particular, w_{ijk} can be fractional, and in
    addition a field can be assigned partially to one cadence and partially
    to another. The correct way to solve this problem is to cast this
    problem as an integer programming problem, where w_{ijk} must be
    integers and we also put a requirement that only one cadence j has
    non-zero w_ijk for any field i.

    However, for problems of the size we have here, the integer
    programming problem is too big. And the lore in this area is that
    the linear programming solution is often very close to the integer
    solution. Because the linear programming maximum is always on a
    boundary, it often lands on a corner of the constraints, which
    is often an integer.

    Nevertheless, for those cases where that is not true, we do the
    following.  For each field i, we sum up the linear programming
    allocations for each cadence j::

        C_ij = \\sum_k w_ijk.

    Then for each field i we pick randomly the cadence according to
    probabilities defined by::

        P_ij = C_ij / \\sum_j C_ij

    Then, we look at the total allocation of the field relative to the
    needed allocation for the cadence::

        \\sum_{jk} w_ijk / A_ij

    If this is greater than one, we accept the field, and if it is
    less than one we take the field with probability equal to the above
    ratio.

    The construct() and solve() methods can also be used to decide
    on time allocation at a fixed choice of cadences per field. This
    can be done as a second step to the solution. The calling sequence
    is::

      allocate.construct()
      allocate.solve()
      allocate.construct(fix_cadence=True)
      allocate.solve(minimize_time=True)

    The first rounds stores its results in the attribute field_array.
    The second round fixes the cadences to those stored in the
    attribute field_array, and minimizes the total time taken. This
    matters mostly if the problem is undersubscribed, so it will
    minimize airmass.
"""
    def __init__(self, slots=None, fields=None, field_slots=None,
                 field_options=None, seed=100, filename=None,
                 observatory=None, cartons=None, cadences=None,
                 observe_all_fields=[], observe_all_cadences=[],
                 dark_prefer=1., minimum_ntargets={}, epoch_overhead=None,
                 verbose=False):
        self.epoch_overhead = None
        self.verbose = verbose
        if(filename is None):
            self.slots = slots
            self.fields = fields
            self.field_slots = field_slots
            self.field_options = field_options
            self.cartons = cartons
            self.cadences = cadences
            self.nfields = len(self.field_options)
            self.epoch_overhead = epoch_overhead
        else:
            self.fromfits(filename=filename)
        if(self.epoch_overhead is None):
            self.epoch_overhead = 5. / 60.
        self.minimum_ntargets = minimum_ntargets
        self.seed = seed
        np.random.seed(self.seed)
        self.observer = scheduler.Observer(observatory=observatory)
        self.observe_all_fields = observe_all_fields
        self.observe_all_cadences = observe_all_cadences
        self.cadencelist = rcadence.CadenceList()
        self.dark_prefer = dark_prefer
        return

    def duration_scale(self, cadence=None, skybrightness=None):
        """Scale duration appropriately given multiexposure epochs
        
        Parameters
        ----------

        cadence : Cadence name
            appropriate cadence

        skybrightness : np.float32
            sky brightness to assume

        Notes
        -----

        Assumes hard-coded dark time of skybrightness < 0.35.

        Allows for different exposure time in dark and bright time.

        The math here is a bit roundabout because the structure of the code.
"""
        cobj = self.cadencelist.cadences[cadence]
        
        scale = 1.
        if(skybrightness < 0.35):
            isky = 0
        else:
            isky = 1
        exptime = self.slots.exptimes[isky]
        total = ((self.epoch_overhead * cobj.nepochs) +
                 (self.slots.exposure_overhead + exptime) *
                  cobj.nexp_total)
        scale = total / (self.slots.durations[isky] * cobj.nexp_total)
        return(scale)

    def xfactor(self, racen=None, deccen=None, skybrightness=None,
                lst=None, cadence=None):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal

        Notes
        -----

        This function sets the cost of observing at high airmass.
        Right now it just scales as airmass**2. Note the cost as a
        function of airmass depends on the targets you are most
        interested in ; since infrared spectra are barely affected by
        transparency but the UV end of the optical spectra affected a
        lot.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        xfactor = airmass**2
        xfactor = xfactor * self.duration_scale(cadence=cadence,
                                                skybrightness=skybrightness)
        return(xfactor)

    def construct(self, fix_cadence=False):
        """Construct the allocinfo attribute with the problem definition

        Parameters
        ----------

        fix_cadence : bool
            If True, use already-determined field cadences (default False)

        Notes
        ------

        The allocinfo attribute is a fairly complicated dictionary
        of dictionaries that contains the information necessary to
        construct the linear programming problem.
"""
        self.allocinfo = collections.OrderedDict()
        self.some_filled = dict()
        self.ncadences = dict()
        fieldids = np.unique(self.field_slots['fieldid'])
        for fieldid in fieldids:
            icurr = np.where(self.field_options['fieldid'] == fieldid)[0]
            self.ncadences[fieldid] = len(icurr)
            curr_slots = self.field_slots[icurr]
            curr_options = self.field_options[icurr]

            curr_cadences = np.array([x.strip()
                                      for x in curr_slots['cadence']])

            if(fix_cadence):
                ifield = np.where(self.field_array['fieldid'] == fieldid)[0]
                fcadence = self.field_array['cadence'][ifield][0]
                if(fcadence == 'none'):
                    self.allocinfo[fieldid] = collections.OrderedDict()
                    continue
                ic = np.where(curr_cadences == fcadence)[0]
                if(len(ic) == 0):
                    print("Inconsistency; no cadence.")
                    print("curr_cadences = {cc}".format(cc=curr_cadences))
                    print("fcadence = {fc}".format(fc=fcadence))
                    return()
                if(len(ic) > 1):
                    print("Inconsistency; more than one cadence.")
                    return()
                curr_slots = curr_slots[ic]
                curr_options = curr_options[ic]
                curr_cadences = curr_cadences[ic]

            alloc = collections.OrderedDict()
            curr_zip = zip(curr_slots, curr_options, curr_cadences)
            self.some_filled[fieldid] = False
            for curr_slot, curr_option, curr_cadence in curr_zip:
                # Skip not-allowed options
                if(curr_slot['allowed'] == False):
                    continue

                minsb = self.cadencelist.cadences[curr_cadence].skybrightness.min()
                if(minsb < 1.):
                    prefer = self.dark_prefer
                else:
                    prefer = 1.
                alloc[curr_cadence] = collections.OrderedDict()
                alloc[curr_cadence]['slots'] = curr_slot['slots'].copy()
                alloc[curr_cadence]['vars'] = ([0] *
                                               (self.slots.nskybrightness *
                                                self.slots.nlst))
                alloc[curr_cadence]['needed'] = float(curr_slot['needed'].copy())
                alloc[curr_cadence]['needed_sb'] = [float(n) for n in curr_slot['needed_sb']]
                alloc[curr_cadence]['filled_sb'] = [float(n) for n in curr_slot['filled_sb']]
                alloc[curr_cadence]['filled'] = curr_slot['filled_sb'].sum()
                alloc[curr_cadence]['allocated_exposures_done'] = curr_slot['allocated_exposures_done']
                alloc[curr_cadence]['original_exposures_done'] = curr_slot['original_exposures_done']
                alloc[curr_cadence]['filled'] = curr_slot['filled_sb'].sum()

                # If there have been SOME observations of this field, note
                # that so we can reduce the field minimum to zero if necessary
                if(alloc[curr_cadence]['filled']):
                    self.some_filled[fieldid] = True

                alloc[curr_cadence]['ngot_pct'] = curr_option['ngot_pct'].copy()
                alloc[curr_cadence]['value'] = float(curr_option['valuegot'] *
                                                     prefer)


            self.allocinfo[fieldid] = alloc
        return()

    def solve(self, minimize_time=False, epsprime=1.e-3, epsilon=1.e-2):
        """Solve the linear programming problem to allocate fields

        Parameters
        ----------

        minimize_time : bool
            If True, minimize time and force all field-cadences (default False)

        Notes
        ------

        Solves the linear programming problem.
"""
        ftype = np.array([x.strip() for x in self.fields['type']])
        field_minimum_float = dict()
        for indx in np.arange(len(self.fields)):
            cftype = ftype[indx]
            cfieldid = self.fields['fieldid'][indx]
            if(minimize_time):
                ifield = np.where(self.field_array['fieldid'] == cfieldid)[0]
                if((len(ifield) == 0)):
                    field_minimum_float[cfieldid] = 0.0
                elif (self.field_array['cadence'][ifield[0]].strip() == 'none'):
                    field_minimum_float[cfieldid] = 0.0
                else:
                    field_minimum_float[cfieldid] = 0.95
            else:
                if(cftype in self.observe_all_fields):
                    # If some of the field has been filled, give the option to
                    # do nothing, unless there is only one cadence to choose from.
                    #if(self.some_filled[cfieldid] & (self.ncadences[cfieldid] > 1)):
                    #field_minimum_float[cfieldid] = 0.
                    #else:
                    field_minimum_float[cfieldid] = 1. - epsilon
                else:
                    field_minimum_float[cfieldid] = 0.

        total = self.slots.slots * self.slots.fclear
        total[:, 0] = total[:, 0] / self.slots.durations[0]
        total[:, 1] = total[:, 1] / self.slots.durations[1]

        solver = pywraplp.Solver("allocate_lst",
                                 pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        solver.SetNumThreads(64)

        # Set up variables; these variables ('vars') will correspond to
        # the number of exposures in each slot for each field-cadence.
        objective = solver.Objective()
        for fieldid in self.allocinfo:
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                ccadence['nvars'] = 0
                for ilst in np.arange(self.slots.nlst, dtype=np.int32):
                    for iskybrightness in np.arange(self.slots.nskybrightness, dtype=np.int32):
                        name = "{f}-{c}-{slst}-{ssb}".format(f=fieldid,
                                                             c=cadence,
                                                             slst=ilst,
                                                             ssb=iskybrightness)
                        if(ccadence['slots'][ilst, iskybrightness] > 0):
                            # Push upper limit so that zero isn't squeezed out
                            var = solver.NumVar(0.0,
                                                ccadence['needed'] - ccadence['filled']
                                                + epsprime,
                                                name)
                            if(minimize_time is False):
                                objective.SetCoefficient(var,
                                                         ccadence['value'] /
                                                         ccadence['needed'])
                            ccadence['vars'][ilst * self.slots.nskybrightness +
                                             iskybrightness] = var
                            ccadence['nvars'] = ccadence['nvars'] + 1

        # Cadences have limits, which limit the total number of exposures
        # in the cadence to the total needed, and as needed in each lunation.
        cadence_constraints = []
        for fieldid in self.allocinfo:
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                if(cadence in self.observe_all_cadences):
                    minval = float(ccadence['needed'] * 0.95 - ccadence['filled'])
                else:
                    minval = float(0.)
                cadence_constraint = solver.Constraint(minval,
                                                       ccadence['needed'] - ccadence['filled']
                                                       + epsprime)

                # set total cadence constraint
                for iskybrightness in np.arange(self.slots.nskybrightness, dtype=np.int32):
                    for ilst in np.arange(self.slots.nlst, dtype=np.int32):
                        if(ccadence['slots'][ilst, iskybrightness] > 0):
                            var = ccadence['vars'][ilst *
                                                   self.slots.nskybrightness +
                                                   iskybrightness]
                            cadence_constraint.SetCoefficient(var, 1.)
                cadence_constraints.append(cadence_constraint)

                # set ratio of dark and bright so there is enough dark
                Asb = np.array([(ccadence['needed_sb'][1] - ccadence['filled_sb'][1]),
                                - (ccadence['needed_sb'][0] - ccadence['filled_sb'][0])])
                solver_inf = solver.infinity()
                min_constraint_sb = - epsprime

                cadence_constraint_sb = solver.Constraint(min_constraint_sb, solver_inf)
                for iskybrightness in np.arange(self.slots.nskybrightness, dtype=np.int32):
                    for ilst in np.arange(self.slots.nlst, dtype=np.int32):
                        if(ccadence['slots'][ilst, iskybrightness] > 0):
                            var = ccadence['vars'][ilst *
                                                   self.slots.nskybrightness +
                                                   iskybrightness]
                            cadence_constraint_sb.SetCoefficient(var, Asb[iskybrightness])
                cadence_constraints.append(cadence_constraint_sb)

        # Constraints on numbers
        mint_constraints = dict()
        mint_index = dict()
        for cname in self.minimum_ntargets:
            index = np.where(self.cartons == cname)[0]
            mint_constraints[cname] = solver.Constraint(float(self.minimum_ntargets[cname]),
                                                        float(10000000.))
            mint_index[cname] = index[0]

        # Maximum of one cadence per field. Note that because this is an
        # LP and not an integer problem, this constraint involves the
        # definition of "fractional" cadences. From the LP solution,
        # we will pick the largest value cadence.
        field_constraints = []
        for fieldid in self.allocinfo:
            if(len(self.allocinfo[fieldid]) > 0):
                field_constraint = solver.Constraint(field_minimum_float[fieldid],
                                                     1. + epsilon)
                for cadence in self.allocinfo[fieldid]:
                    ccadence = self.allocinfo[fieldid][cadence]

                    invneeded = 1. / (ccadence['needed'] - ccadence['filled'] + epsprime)

                    for ilst in np.arange(self.slots.nlst, dtype=np.int32):
                        for iskybrightness in np.arange(self.slots.nskybrightness, dtype=np.int32):
                            if(ccadence['slots'][ilst, iskybrightness] > 0):
                                var = ccadence['vars'][ilst * self.slots.nskybrightness +
                                                       iskybrightness]
                                field_constraint.SetCoefficient(var, invneeded)
                                for cname in self.minimum_ntargets:
                                    imint = mint_index[cname]
                                    mint_constraints[cname].SetCoefficient(
                                        var, invneeded *
                                        float(ccadence['ngot_pct'][imint]))
                field_constraints.append(field_constraint)
            else:
                if(field_minimum_float[fieldid] > 0.):
                    raise ValueError("No allowed cadences for field with required observations")

        # Constrain sum of each slot to be less than total. Here the
        # units are still in numbers of exposures, but we multiply by
        # a scaling factor (xfactor) that depends on airmass to account
        # for the cost of high airmass observations.
        slot_constraints = [[0] * self.slots.nskybrightness] * self.slots.nlst
        for ilst in np.arange(self.slots.nlst, dtype=np.int32):
            for iskybrightness in np.arange(self.slots.nskybrightness, dtype=np.int32):
                slot_constraints[ilst][iskybrightness] = solver.Constraint(0., float(total[ilst, iskybrightness]))
                for fieldid in self.allocinfo:
                    ifield = np.where(fieldid == self.fields['fieldid'])[0]
                    field_racen = self.fields['racen'][ifield]
                    field_deccen = self.fields['deccen'][ifield]
                    for cadence in self.allocinfo[fieldid]:
                        ccadence = self.allocinfo[fieldid][cadence]
                        if(ccadence['slots'][ilst, iskybrightness] > 0):
                            xfactor = self.xfactor(racen=field_racen,
                                                   deccen=field_deccen,
                                                   cadence=cadence,
                                                   skybrightness=self.slots.skybrightness[iskybrightness + 1],
                                                   lst=self.slots.lst[ilst],
                                                   filled_sb=np.array(ccadence['filled_sb']))
                            slot_constraints[ilst][iskybrightness].SetCoefficient(ccadence['vars'][ilst * self.slots.nskybrightness + iskybrightness], float(xfactor))
                            if(minimize_time is True):
                                objective.SetCoefficient(ccadence['vars'][ilst * self.slots.nskybrightness + iskybrightness], float(xfactor))
            
        # Solve the problem
        if(minimize_time is True):
            objective.SetMinimization()
        else:
            objective.SetMaximization()

        status = solver.Solve()
        if(status != 0):
            print("Solver failed: {status}.".format(status=status))
            return(status)
        else:
            print("Solver succeeded.")

        # Extract the solution.
        # Here var is a number of exposures, and so is allocation.
        tval = 0.
        for fieldid in self.allocinfo:
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                ccadence['allocation'] = np.zeros((self.slots.nlst,
                                                   self.slots.nskybrightness),
                                                  dtype=np.float32)
                for ilst in np.arange(self.slots.nlst, dtype=np.int32):
                    for iskybrightness in np.arange(self.slots.nskybrightness, dtype=np.int32):
                        if(ccadence['slots'][ilst, iskybrightness] > 0):
                            var = ccadence['vars'][ilst * self.slots.nskybrightness + iskybrightness]
                            ccadence['allocation'][ilst, iskybrightness] = var.solution_value()
                            cval = var.solution_value() * ccadence['value'] / ccadence['needed']
                            tval = tval + cval

        # Decide on which cadences to pick.
        field_array_dtype = [('fieldid', np.int32),
                             ('racen', np.float64),
                             ('deccen', np.float64),
                             ('cadence', np.unicode_, 30),
                             ('nallocated', np.int32),
                             ('nallocated_sb', np.int32, self.slots.nskybrightness),
                             ('nallocated_full', np.int32),
                             ('nallocated_full_sb', np.int32, self.slots.nskybrightness),
                             ('needed', np.int32),
                             ('needed_sb', np.int32, self.slots.nskybrightness),
                             ('filled', np.int32),
                             ('filled_sb', np.int32, self.slots.nskybrightness),
                             ('allocated_exposures_done', np.int32, 100),
                             ('original_exposures_done', np.int32, 100),
                             ('xfactor', np.float32,
                              (self.slots.nlst, self.slots.nskybrightness)),
                             ('slots_exposures', np.float32,
                              (self.slots.nlst, self.slots.nskybrightness)),
                             ('slots_time', np.float32,
                              (self.slots.nlst, self.slots.nskybrightness))]
        field_array = np.zeros(len(self.allocinfo), dtype=field_array_dtype)
        field_array['allocated_exposures_done'] = -1
        field_array['original_exposures_done'] = -1
        for findx, fieldid in zip(np.arange(len(self.allocinfo)),
                                  self.allocinfo):
            field_array['fieldid'][findx] = fieldid
            ifield = np.where(fieldid == self.fields['fieldid'])[0]
            field_array['racen'][findx] = self.fields['racen'][ifield]
            field_array['deccen'][findx] = self.fields['deccen'][ifield]
            ncadence = len(self.allocinfo[fieldid])
            cadence_totals = np.zeros(ncadence, dtype=np.float32)
            slots_totals = np.zeros((self.slots.nlst,
                                     self.slots.nskybrightness), dtype=np.float32)
            for indx, cadence in zip(np.arange(ncadence),
                                     self.allocinfo[fieldid]):
                ccadence = self.allocinfo[fieldid][cadence]
                print("fieldid {fid} {c} {t}".format(fid=fieldid, c=cadence,
                                                     t=ccadence['allocation'].sum()),
                      flush=True)
                slots_totals = slots_totals + ccadence['allocation']
                cadence_totals[indx] = ccadence['allocation'].sum()
            print("fieldid {fid} {s}".format(fid=fieldid, s=slots_totals),
                  flush=True)
            field_total = cadence_totals.sum()
            field_array['cadence'][findx] = 'none'
            field_array['slots_exposures'][findx] = (
                np.zeros((self.slots.nlst, self.slots.nskybrightness),
                         dtype=np.float32))
            field_array['slots_time'][findx] = (
                np.zeros((self.slots.nlst, self.slots.nskybrightness),
                         dtype=np.float32))
            if(field_total > 0.):
                cadence_totals = cadence_totals / field_total
                cadence_cumulative = cadence_totals.cumsum()
                choose = np.random.random()
                icadence = np.where(cadence_cumulative > choose)[0][0]
                cadence = list(self.allocinfo[fieldid].keys())[icadence]
                ii = np.where(cadence_totals > 0)[0]
                if((len(ii) > 1) & (cadence_totals.max() < 0.9)):
                    print("Multiples for f={fid}".format(fid=fieldid))
                    print(self.allocinfo[fieldid].keys())
                    print(cadence_totals)
                    print(cadence_cumulative)
                    print(choose)
                    print(icadence)
                    print(cadence)
                field_array['cadence'][findx] = cadence
                field_array['needed'][findx] = self.allocinfo[fieldid][cadence]['needed']
                field_array['needed_sb'][findx] = self.allocinfo[fieldid][cadence]['needed_sb']
                field_array['filled_sb'][findx] = self.allocinfo[fieldid][cadence]['filled_sb']
                field_array['filled'][findx] = self.allocinfo[fieldid][cadence]['filled']
                field_array['allocated_exposures_done'][findx] = self.allocinfo[fieldid][cadence]['allocated_exposures_done']
                field_array['original_exposures_done'][findx] = self.allocinfo[fieldid][cadence]['original_exposures_done']
                slots_totals_total = slots_totals.sum()
                if(slots_totals_total > 0):
                    normalize = ((field_array['needed_sb'][findx, :].sum() -
                                  field_array['filled_sb'][findx, :].sum()) /
                                 slots_totals_total)
                    field_array['slots_exposures'][findx, :, :] = (slots_totals * normalize)
                else:
                    print("Something weird! slots_totals_total should not be zero here", flush=True)

            field_array['nallocated'][findx] = np.int32(
                field_array['slots_exposures'][findx, :, :].sum() + 0.001)
            for isky in np.arange(self.slots.nskybrightness, dtype=np.int32):
                field_array['nallocated_sb'][findx, isky] = np.int32(
                    field_array['slots_exposures'][findx, :, isky].sum() + 0.001)
            field_array['nallocated_full'][findx] = (field_array['nallocated'][findx] +
                                                     field_array['filled'][findx])
            for isky in np.arange(self.slots.nskybrightness, dtype=np.int32):
                field_array['nallocated_full_sb'][findx, isky] = (field_array['nallocated_sb'][findx, isky] +
                                                                  field_array['filled_sb'][findx, isky])

        fscadence = np.array([x.strip()
                              for x in self.field_slots['cadence']])
        for findx in np.arange(len(field_array), dtype=np.int32):
            field = field_array[findx]
            fcadence = field['cadence'].strip()
            if(fcadence != 'none'):
                islots = np.where((self.field_slots['fieldid'] ==
                                   field['fieldid']) &
                                  (fscadence == fcadence))[0][0]
                curr_slots = self.field_slots[islots]['slots']
                curr_filled_sb = self.field_slots[islots]['filled_sb']
                for ilst in np.arange(self.slots.nlst, dtype=np.int32):
                    lst = self.slots.lst[ilst]
                    for isb in np.arange(self.slots.nskybrightness,
                                         dtype=np.int32):
                        skybrightness = self.slots.skybrightness[isb + 1]
                        if(curr_slots[ilst, isb] > 0):
                            xfactor = self.xfactor(racen=field['racen'],
                                                   deccen=field['deccen'],
                                                   cadence=fcadence,
                                                   skybrightness=skybrightness,
                                                   lst=lst,
                                                   filled_sb=curr_filled_sb)
                            field['slots_time'][ilst, isb] = field['slots_exposures'][ilst, isb] * xfactor * self.slots.durations[isb]
                            field['xfactor'][ilst, isb] = xfactor

        print(field_array['slots_exposures'].sum(axis=0))
        print(field_array['slots_time'].sum(axis=0))
        print(self.slots.durations)
        print(total)

        self.field_array = field_array

        return(status)

    def tofits(self, filename=None):
        """Write field allocation array to a FITS file

        Parameters
        ----------

        filename : str
            file name to write to

        Notes
        ------

        Writes all array attributes as a binary table.
"""
        hdr = robostrategy.header.rsheader()
        hdr.append({'name':'EPOVER',
                    'value':self.epoch_overhead,
                    'comment':'Epoch overhead assumed (hours)'})
        fitsio.write(filename, self.field_array, header=hdr,
                     clobber=True, extname='ALLOCATE')
        self.slots.tofits(filename=filename, clobber=False)
        fitsio.write(filename, self.fields, extname='FIELDS', clobber=False)
        fitsio.write(filename, self.field_slots, extname='FSLOTS', clobber=False)
        fitsio.write(filename, self.field_options, extname='OPTIONS', clobber=False)
        cartons_arr = np.zeros(len(self.cartons),
                               dtype=[('carton', 'U50')])
        cartons_arr['carton'] = self.cartons
        cadences_arr = np.zeros(len(self.cadences),
                               dtype=[('cadences', 'U50')])
        cadences_arr['cadences'] = self.cadences
        fitsio.write(filename, cartons_arr, extname='CARTONS', clobber=False)
        fitsio.write(filename, cadences_arr, extname='CADENCES', clobber=False)
        return

    def fromfits(self, filename=None):
        """Read field allocation array from a FITS file

        Parameters
        ----------

        filename : str
            file name to write to

        Notes
        ------

        Reads all attributes from a binary FITS table.
"""
        self.field_array, hdr = fitsio.read(filename, header=True, ext=1)
        if('EPOVER' in hdr):
            self.epoch_overhead = np.float32(hdr['EPOVER'])
        self.slots = robostrategy.slots.Slots()
        self.slots.fromfits(filename, ext=2)
        self.fields = fitsio.read(filename, ext=3)
        self.field_slots = fitsio.read(filename, ext=4)
        self.field_options = fitsio.read(filename, ext=5)
        self.cartons = fitsio.read(filename, ext=6)
        return

    def _available_lst(self):
        available = self.slots.slots * self.slots.fclear
        return(available)

    def _used_lst(self):
        used = self.field_array['slots_time'][:, :, :].sum(axis=0)
        return(used)

    def _got_ra(self):
        got = np.zeros((self.slots.nlst, self.slots.nskybrightness),
                       dtype=np.float32)
        for iskybrightness in np.arange(self.slots.nskybrightness):
            ngot = self.field_array['slots_time'][:, :, iskybrightness].sum(axis=1)
            rahist, rabinedges = np.histogram(self.field_array['racen'] / 15.,
                                              range=[0., 24.],
                                              weights=ngot,
                                              bins=self.slots.nlst)
            got[:, iskybrightness] = rahist

        return(got)

    def _used_lst_cadence(self, iskybrightness=None):
        used = np.zeros(self.field_array['slots_time'][0, :, 0].shape,
                        dtype=np.float32)
        for ifield, field in enumerate(self.field_array):
            if(field['cadence'] == 'none'):
                continue

            issky = (self.cadencelist.cadences[field['cadence']].skybrightness
                     == self.slots.skybrightness[iskybrightness + 1])
            nsky = (field['needed_sb'][iskybrightness].sum() -
                    field['filled_sb'][iskybrightness].sum())

            if(nsky > 0):
                avgx = ((field['xfactor'] * field['slots_time']).sum() /
                        field['slots_time'].sum())
                skytime = nsky * self.slots.durations[iskybrightness] * avgx
                curr_used = field['slots_time'][:, iskybrightness]
                if(curr_used.sum() > 0):
                    curr_used = skytime * curr_used / curr_used.sum()
                used = used + curr_used

        return(used)

    def plot_lst(self, iskybrightness=None, title=None, loc=2):
        """Plot the LST distributions for the allocations

        Parameters
        ----------

        iskybrightness : ndarray of np.int32
            indices of the sky brightness classes to plot

        title : str
            title to put on plot
"""

        available = self._available_lst()
        used = self._used_lst()
        got = self._got_ra()
        useddark = None

        if(iskybrightness is None):
            used = used.sum(axis=1)
            available = available.sum(axis=1)
            got = got.sum(axis=1)
        else:
            used = used[:, iskybrightness]
            available = available[:, iskybrightness]
            got = got[:, iskybrightness]
            if(iskybrightness == 0):
                useddark = self._used_lst_cadence(iskybrightness=0)

        plt.plot(used, color='red', linewidth=3, alpha=0.6,
                 label='Hours used per LST ({t:>3.1f} h)'.format(t=used.sum()))
        plt.plot(available, color='red', linewidth=1,
                 label='Hours available per LST ({t:>.1f} h)'.format(t=available.sum()))
        print(used.sum())
        print(available.sum())
        plt.plot(got, color='blue', linewidth=3, alpha=0.6,
                 label='Hours observed per RA ({t:>.1f} h)'.format(t=got.sum()))
        print(got.sum())
        if(useddark is not None):
            plt.plot(useddark, color='green', linewidth=3, alpha=0.6,
                     label='Dark cadence used per LST ({t:>.1f} h)'.format(t=useddark.sum()))
            print(useddark.sum())

        plt.xlabel('LST or RA (hours)')
        plt.ylabel('Exposure hours')
        plt.ylim(np.array([-0.05, 1.5]) * np.array([got.max(), used.max(),
                                                    available.max()]).max())
        plt.legend(loc=loc, fontsize=12)
        if(title is not None):
            plt.title(title)

        return

    def _convert_radec(self, m, ra, dec):
        return m(((360. - ra) + 180.) % 360., dec, inverse=False)

    def plot_fields(self, full=False, indx=None, label=False, linear=False,
                    colorbar=True, lon_0=270., darkorbright=None, vmax=None,
                    vmin=None, **kwargs):
        """Plot the RA/Dec distribution of fields allocated

        Parameters
        ----------

        indx : ndarray of np.int32
            indices of fields to plot
"""

        if(indx is None):
            indx = np.arange(len(self.field_array), dtype=np.int32)

        if basemap is None:
            raise ImportError('basemap was not imported. Is it installed?')

        m = basemap.Basemap(projection='moll', lon_0=lon_0, resolution='c',
                            celestial=True)

        # draw parallels and meridians.
        m.drawparallels(np.arange(-90., 120., 30.),
                        linewidth=0.5,
                        labels=[0, 0, 0, 0],
                        labelstyle='+/-')
        m.drawmeridians(np.arange(0., 420., 60.), linewidth=0.5)
        m.drawmapboundary(fill_color='#f0f0f0')

        if(full):
            nallocated_tag = 'nallocated_full'
            nallocated_sb_tag = 'nallocated_full_sb'
        else:
            nallocated_tag = 'nallocated'
            nallocated_sb_tag = 'nallocated_sb'

        ii = np.where(self.field_array[nallocated_tag][indx] > 0)[0]
        ii = indx[ii]

        nallocated_max = 1
        if(len(ii) > 0):
            isrm = np.array([('x8' in c) for c in self.field_array['cadence'][ii]], dtype=bool)
            ilimit = np.where(isrm == False)[0]
            if(len(ilimit) > 0):
                nallocated_max = self.field_array[nallocated_tag][ii[ilimit]].max()

        if(darkorbright is not None):
            if(darkorbright == 'dark'):
                nallocated = self.field_array[nallocated_sb_tag][ii, 0]
            if(darkorbright == 'bright'):
                nallocated = self.field_array[nallocated_sb_tag][ii, 1]
        else:
            nallocated = self.field_array[nallocated_tag][ii]

        if(linear is False):
            nallocated_value = np.log10(nallocated)
            nallocated_value_max = np.log10(nallocated_max)
            colorbar_label = r'$\log_{10} N$'
            if(vmin is None):
                vmin = -0.1
        else:
            nallocated_value = nallocated
            nallocated_value_max = nallocated_max
            colorbar_label = '$N$'
            if(vmin is None):
                vmin = 0.

        if(vmax is None):
            vmax = nallocated_value_max

        if(label is False):
            (xx, yy) = self._convert_radec(m, self.field_array['racen'][ii],
                                           self.field_array['deccen'][ii])
            plt.scatter(xx, yy, s=4, c=nallocated_value, vmin=vmin, vmax=vmax,
                        cmap='Blues', **kwargs)
            if(colorbar):
                cb = plt.colorbar()
                cb.set_label(colorbar_label)
        else:
            cadences = np.array(["_".join(x.strip().split('-')[0].split('_')[0:-1])
                                 for x in self.field_array['cadence'][ii]])
            ucadences = np.unique(cadences)
            for ucadence in ucadences:
                jj = ii[np.where(cadences == ucadence)[0]]
                (xx, yy) = self._convert_radec(m,
                                               self.field_array['racen'][jj],
                                               self.field_array['deccen'][jj])
                plt.scatter(xx, yy, s=4, cmap='Blues', label=ucadence)

            plt.legend(loc='lower center', fontsize=8, ncol=2)

        return(vmin, vmax)


class AllocateLSTCostA(AllocateLST):
    """LST allocation object for robostrategy, with cost model A

    This class is exactly like AllocateLST, but it models the airmass
    dependence differently. If the maximum sky brightness requirement of the
    cadence is <= 0.4, then it assumes you are working in the optical
    and care about blue throughput, and it scales exposure time with
    airmass^2. If not, it scales the cost as airmass^0.5.
"""
    def xfactor(self, racen=None, deccen=None, skybrightness=None, lst=None,
                cadence=None, filled_sb=np.array([0., 0.], dtype=np.float32)):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        filled_sb : [2] ndarray of np.int32
            number of filled dark and bright exposures

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal

        Notes
        -----

        If the minimum sky brightness requirement of the cadence is <= 0.4,
        then it assumes you are working in the optical and care about blue
        throughput, and it scales exposure time with airmass^2. If not, it
        scales the cost as airmass^0.25.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        exponent = 0.25
        if(self.cadencelist.cadences[cadence].skybrightness.min() < 0.4):
            exponent = 2.0
        xfactor = airmass**exponent
        xfactor = xfactor * self.duration_scale(cadence=cadence,
                                                skybrightness=skybrightness)
        return(xfactor)


class AllocateLSTCostB(AllocateLST):
    """LST allocation object for robostrategy, with cost model B

    This class is exactly like AllocateLSTCostA,
    but with scaling in bright time of airmass^1.
"""
    def xfactor(self, racen=None, deccen=None, skybrightness=None, lst=None,
                cadence=None, filled_sb=np.array([0., 0.], dtype=np.float32)):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        filled_sb : [2] ndarray of np.int32
            number of filled dark and bright exposures

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal

        Notes
        -----

        If the minimum sky brightness requirement of the cadence is <= 0.4,
        then it assumes you are working in the optical and care about blue
        throughput, and it scales exposure time with airmass^2. If not, it
        scales the cost as airmass^1.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        exponent = 1.
        if(self.cadencelist.cadences[cadence].skybrightness.min() < 0.4):
            exponent = 2.0
        xfactor = airmass**exponent
        xfactor = xfactor * self.duration_scale(cadence=cadence,
                                                skybrightness=skybrightness)
        return(xfactor)


class AllocateLSTCostC(AllocateLST):
    """LST allocation object for robostrategy, with cost model C

    This class is exactly like AllocateLSTCostA,
    but with xfactor scaling in dark time of airmass^1.
"""
    def xfactor(self, racen=None, deccen=None, skybrightness=None, lst=None,
                cadence=None, filled_sb=np.array([0., 0.], dtype=np.float32)):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal

        Notes
        -----

        If the minimum sky brightness requirement of the cadence is <= 0.4,
        then it assumes you are working in the optical and care about blue
        throughput, and it scales exposure time with airmass^1. If not, it
        scales the cost as airmass^0.25.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        exponent = 0.25
        if(self.cadencelist.cadences[cadence].skybrightness.min() < 0.4):
            exponent = 1.0
        xfactor = airmass**exponent
        xfactor = xfactor * self.duration_scale(cadence=cadence,
                                                skybrightness=skybrightness)
        return(xfactor)


class AllocateLSTCostD(AllocateLST):
    """LST allocation object for robostrategy, with cost model D

    This class is exactly like AllocateLSTCostD,
    but with xfactor scaling in dark time of airmass^1,
    and in bright time of airmass^0.05.
"""
    def xfactor(self, racen=None, deccen=None, skybrightness=None, lst=None,
                cadence=None, filled_sb=np.array([0., 0.], dtype=np.float32)):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal

        Notes
        -----

        If the minimum sky brightness requirement of the cadence is <= 0.4,
        then it assumes you are working in the optical and care about blue
        throughput, and it scales exposure time with airmass^1. If not, it
        scales the cost as airmass^0.05.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        exponent = 0.05
        if(self.cadencelist.cadences[cadence].skybrightness.min() < 0.4):
            exponent = 1.0
        xfactor = airmass**exponent
        xfactor = xfactor * self.duration_scale(cadence=cadence,
                                                skybrightness=skybrightness)
        return(xfactor)


class AllocateLSTCostE(AllocateLST):
    """LST allocation object for robostrategy, with cost model E

    This class is exactly like AllocateLSTCostD,
    but using the fraction of the cadence in dark and bright
    time to scale things.
"""
    def xfactor(self, racen=None, deccen=None, skybrightness=None, lst=None,
                cadence=None, filled_sb=np.array([0., 0.], dtype=np.float32)):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        filled_sb : [2] ndarray of np.int32
            number of filled dark and bright exposures

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal for this skybrightness

        Notes
        -----

        If the sky brightness requirement of an epoch is <= 0.4,
        then it assumes you are working in the optical and care about blue
        throughput, and it scales exposure time with airmass^1. If not, it
        scales the cost as airmass^0.05. For mixed cadences, it only scales
        the dark fraction as airmass^1.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        exponent_bright = 0.05
        exponent_dark = 1.0
        idark = np.where(self.cadencelist.cadences[cadence].skybrightness < 0.4)[0]
        ibright = np.where(self.cadencelist.cadences[cadence].skybrightness >= 0.4)[0]
        ndark = self.cadencelist.cadences[cadence].nexp[idark].sum()
        nbright = self.cadencelist.cadences[cadence].nexp[ibright].sum()
        if(filled_sb.sum() < self.cadencelist.cadences[cadence].nexp_total):
            fdark = (np.float32(ndark - filled_sb[0]) /
                     np.float32(self.cadencelist.cadences[cadence].nexp_total - filled_sb.sum()))
        else:
            # If all filled up, this really doesn't matter
            fdark = np.float32(ndark) / np.float32(self.cadencelist.cadences[cadence].nexp_total)
        xfactor_dark = (fdark) * (airmass**exponent_dark)
        xfactor_dark = xfactor_dark * self.duration_scale(cadence=cadence,
                                                          skybrightness=0.)
        xfactor_bright = (1 - fdark) * (airmass**exponent_bright)
        xfactor_bright = xfactor_bright * self.duration_scale(cadence=cadence,
                                                              skybrightness=1.)
        xfactor = xfactor_dark + xfactor_bright
        return(xfactor)


class AllocateLSTCostF(AllocateLST):
    """LST allocation object for robostrategy, with cost model F

    This class is exactly like AllocateLSTCostE, but for dark_1x3 we 
    rescale the cost to assume a fourth exposure is necessary.
"""
    def xfactor(self, racen=None, deccen=None, skybrightness=None, lst=None,
                cadence=None, filled_sb=np.array([0., 0.], dtype=np.float32)):
        """Exposure time cost factor relative to nominal

        Parameters
        ----------

        racen : np.float64
            RA center of field, degrees

        deccen : np.float64
            Dec center of field, degrees

        cadence : str
            cadence name

        skybrightness : np.float32
            maximum sky brightness of observation

        lst : np.float32
            LST of observation, hours

        Returns
        -------

        xfactor : np.float64
            length of exposure relative to nominal

        Notes
        -----

        If the sky brightness requirement of an epoch is <= 0.4,
        then it assumes you are working in the optical and care about blue
        throughput, and it scales exposure time with airmass^1. If not, it
        scales the cost as airmass^0.05. For mixed cadences, it only scales
        the dark fraction as airmass^1.
"""
        ha = self.observer.ralst2ha(ra=racen, lst=lst * 15.)
        (alt, az) = self.observer.hadec2altaz(ha=ha, dec=deccen,
                                              lat=self.observer.latitude)
        airmass = self.observer.alt2airmass(alt=alt)
        exponent_bright = 0.05
        exponent_dark = 1.0
        idark = np.where(self.cadencelist.cadences[cadence].skybrightness < 0.4)[0]
        fdark = (np.float32(len(idark) - filled_sb[0])
                 / np.float32(self.cadencelist.cadences[cadence].nepochs))
        xfactor = (fdark * airmass**exponent_dark +
                   (1 - fdark) * airmass**exponent_bright)
        if(cadence == 'dark_1x3'):
            xfactor = xfactor * 4. / 3.
        xfactor = xfactor * self.duration_scale(cadence=cadence,
                                                skybrightness=skybrightness)
        return(xfactor)
