# @Author: Michael R. Blanton
# @Date: Aug 3, 2018
# @Filename: field_assign_gg
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
from ortools.linear_solver import pywraplp


class AllocateLST(object):
    """LST allocation object for robostrategy

    Parameters:
    ----------

    slots : robostrategy.slots.Slots object
        information about available time in survey

    field_slots : ndarray
        for each field, which slots are available

    field_options : ndarray
        for each field, what cadence options to choose from

    seed : int, np.int32
        random seed for solution selection (default 100)

    Attributes:
    ----------

    slots : robostrategy.slots.Slots object
        information about available time in survey

    field_slots : ndarray
        for each field, which slots are available

    field_options : ndarray
        for each field, what cadence options to choose from

    nfields : int
        number of fields

    seed : int, np.int32
        random seed for solution selection

    allocinfo : OrderedDict
        complicated object storing linear programming variables
        (set by construct() method)

    field_array : ndarray
        for each field, solution as to how to observe it, with fields:
         'fieldid' - field identifier
         'cadence' - cadence name
         'nfilled' - number of exposures allocated
         'slots' - exposures allocated for each slot
        (set by solve() method)

    Methods:
    -------

    construct() : create allocinfo attribute organizing field info
    solve() : solve for allocation, create field_array attribute with results
    tofits() : write field_array to a FITS file
    plot_full() : plot the LST distribution of field_array

    Notes:
    -----

    This class applies a linear programming approach to allocating 
    field observations.

    The inputs are:
      - the available time as a function of LST and lunation (slots)
      - the slots that each field could be observed in (field_slots)
      - the cadences available for each field (field_options)

    The code constructs a linear programming problem of the following
    form. We define a set of variables indexed w_ijk, where i indexes
    the fields, j indexes the cadences available for each field, and k
    indexes the usable slots for that field and cadence (which depend
    on LST and lunation). There are not necessarily the same number of
    indices for each field.

    We then impose the following constraints:

      \sum_{ij} w_{ijk} < T_k, with T_k the maximum # exposures in each slot
      0 <= w_{ijk} <= A_{ij}, with A_{ij} the total time necessary for the
                              field-cadence i-j
      \sum_k w_{ijk} <= A_{ij}
      \sum_{jk} (w_{ijk} / A_{ij}) <= 1, which means you can only have
          the equivalent of a single cadence assigned for a field.

    Then we define a value:

      V = \sum_{ijk} (w_{ijk} / A_{ij}) V_{ij}

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

    Nevertheless, for those cases where that is not true, 

"""
    def __init__(self, slots=None, field_slots=None, field_options=None,
                 seed=100):
        self.slots = slots
        self.field_slots = field_slots
        self.field_options = field_options
        self.nfields = len(self.field_options)
        self.seed = seed
        np.random.seed(self.seed)
        return

    def construct(self):
        """Construct the allocinfo attribute with the problem definition

        Notes:
        ------

        The allocinfo attribute is a fairly complicated dictionary
        of dictionaries that contains the information necessary to
        construct the linear programming problem.
"""
        for field_option, field_slot in zip(self.field_options,
                                            self.field_slots):
            alloc = collections.OrderedDict()
            if(field_option['ngot'].max() == 0):
                cadences = [1]
            else:
                fgot = field_option['ngot'] / field_option['ngot'].max()
                fgot_unique, iunique = np.unique(fgot, return_index=True)
                indx = np.where(fgot_unique > 0.75)[0]
                if(len(indx) > 0):
                    cadences = [int(iunique[i] + 1) for i in indx]
                else:
                    cadences = [int(iunique[-1] + 1)]

            for cadence in cadences:
                alloc[cadence] = collections.OrderedDict()
                alloc[cadence]['slots'] = field_slot['slots']
                alloc[cadence]['vars'] = [0] * (self.slots.nlunation * self.slots.nlst)
                alloc[cadence]['needed'] = float(cadence)
                alloc[cadence]['value'] = float(cadence)

            self.allocinfo[field_option['fieldid']] = alloc
        return()

    def solve(self):
        """Solve the linear programming problem to allocate fields

        Notes:
        ------

        Solves the linear programming problem.
"""
        total = self.slots.slots / self.slots.duration * self.slots.fclear

        solver = pywraplp.Solver("allocate_lst",
                                 pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

        # Set up variables
        objective = solver.Objective()
        for fieldid in self.allocinfo:
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                ccadence['nvars'] = 0
                for ilst in range(self.slots.nlst):
                    for ilunation in range(self.slots.nlunation):
                        name = "{f}-{c}-{slst}-{slun}".format(f=fieldid,
                                                              c=cadence,
                                                              slst=ilst,
                                                              slun=ilunation)
                        if(ccadence['slots'][ilst, ilunation]):
                            var = solver.NumVar(0.0, ccadence['needed'], name)
                            objective.SetCoefficient(var,
                                                     ccadence['value'] /
                                                     ccadence['needed'])
                            ccadence['vars'][ilst * self.slots.nlunation +
                                             ilunation] = var
                            ccadence['nvars'] = ccadence['nvars'] + 1

        # Cadences have limits
        cadence_constraints = []
        for fieldid in self.allocinfo:
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                cadence_constraint = solver.Constraint(0., ccadence['needed'])
                for ilst in range(self.slots.nlst):
                    for ilunation in range(self.slots.nlunation):
                        if(ccadence['slots'][ilst, ilunation]):
                            var = ccadence['vars'][ilst * self.slots.nlunation +
                                                   ilunation]
                            cadence_constraint.SetCoefficient(var, 1.)
            cadence_constraints.append(cadence_constraint)

        # Only one cadence per field
        field_constraints = []
        for fieldid in self.allocinfo:
            field_constraint = solver.Constraint(0., 1.)
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                for ilst in range(self.slots.nlst):
                    for ilunation in range(self.slots.nlunation):
                        if(ccadence['slots'][ilst, ilunation]):
                            var = ccadence['vars'][ilst * self.slots.nlunation +
                                                   ilunation]
                            invneeded = 1. / ccadence['needed']
                            field_constraint.SetCoefficient(var, invneeded)
            field_constraints.append(field_constraint)

        # Constrain sum of each slot to be less than total
        slot_constraints = [[0] * self.slots.nlunation] * self.slots.nlst
        for ilst in range(self.slots.nlst):
            for ilunation in range(self.slots.nlunation):
                slot_constraints[ilst][ilunation] = solver.Constraint(0., float(total[ilst, ilunation]))
                for fieldid in self.allocinfo:
                    for cadence in self.allocinfo[fieldid]:
                        ccadence = self.allocinfo[fieldid][cadence]
                        if(ccadence['slots'][ilst, ilunation]):
                            slot_constraints[ilst][ilunation].SetCoefficient(ccadence['vars'][ilst * self.slots.nlunation + ilunation], 1.)

        # Solve the problem
        objective.SetMaximization()
        status = solver.Solve()

        # Extract the solution
        for fieldid in self.allocinfo:
            for cadence in self.allocinfo[fieldid]:
                ccadence = self.allocinfo[fieldid][cadence]
                ccadence['allocation'] = np.zeros((self.slots.nlst,
                                                   self.slots.nlunation),
                                                  dtype=np.float32)
                for ilst in range(self.slots.nlst):
                    for ilunation in range(self.slots.nlunation):
                        if(ccadence['slots'][ilst, ilunation]):
                            var = ccadence['vars'][ilst * self.slots.nlunation + ilunation]
                            ccadence['allocation'][ilst, ilunation] = var.solution_value()

        # Decide on which cadences to pick.
        field_array_dtype = [('fieldid', np.int32),
                             ('cadence', 'a30'),
                             ('nfilled', np.int32),
                             ('slots', np.float32, (self.slots.nlst,
                                                    self.slots.nlunation))]
        field_array = np.zeros(len(self.allocinfo), dtype=field_array_dtype)
        for findx, fieldid in zip(np.arange(len(self.allocinfo)),
                                  self.allocinfo):
            ncadence = len(self.allocinfo[fieldid])
            cadence_totals = np.zeros(ncadence, dtype=np.float32)
            slots_totals = np.zeros((self.slots.nlst,
                                     self.slots.nlunation), dtype=np.float32)
            for indx, cadence in zip(np.arange(ncadence),
                                     self.allocinfo[fieldid]):
                ccadence = self.allocinfo[fieldid][cadence]
                slots_totals = slots_totals + ccadence['allocation']
                cadence_totals[indx] = ccadence['allocation'].sum()
            field_total = cadence_totals.sum()
            field_array['cadence'][findx] = '0'
            field_array['slots'][fieldid] = self.allocinfo[fieldid][cadence]['slots'] * 0.
            if(field_total > 0.):
                cadence_totals = cadence_totals / field_total
                cadence_cumulative = cadence_totals.cumsum()
                choose = np.random.random()
                icadence = np.where(cadence_cumulative > choose)[0][0]
                cadence = list(self.allocinfo[fieldid].keys())[icadence]
                choose_field = np.random.random()
                field_array['fieldid'][findx] = fieldid
                if(choose_field <
                   field_total / self.allocinfo[fieldid][cadence]['needed']):
                    field_array['cadence'][findx] = str(cadence)
                    field_array['slots'][findx] = slots_totals

            field_array['nfilled'][findx] = field_array['slots'][findx, :, :].sum()

        self.field_array = field_array

        return(status)

    def tofits(self, filename=None):
        """Write field allocation array to a FITS file

        Parameters:
        ----------

        filename : str
            file name to write to

        Notes:
        ------

        Writes the field_array attribute as a binary table.
"""
        fitsio.write(filename, self.field_array, clobber=True)
        return

    def plot_full(self):
        """Plot the LST distributions for the allocations
"""
        total = self.slots.slots / self.slots.duration * self.slots.fclear
        used = self.field_array['slots'][:, :, :].sum(axis=0).sum(axis=1)
        weights = np.array([int(c.decode().strip()) for c in
                            self.field_array['cadence']])
        #color = ['red', 'blue']
        #for ilun in range(self.slots.nlunation):
        #    plt.plot(used[:, ilun], color=color[ilun], linewidth=3, alpha=0.6)
        plt.plot(used[:], color='red', linewidth=3, alpha=0.6)
        plt.plot(total.sum(axis=1), color='red', linewidth=1)
        print(used.sum())
        rahist, rabinedges = np.histogram(self.field_options['racen'] / 15.,
                                          range=[0., 24.],
                                          weights=self.field_array['nfilled'],
                                          bins=24)
        plt.plot(rahist, color='blue', linewidth=3, alpha=0.6)
        print(rahist.sum())
        rahist, rabinedges = np.histogram(self.field_options['racen'] / 15.,
                                          range=[0., 24.], weights=weights,
                                          bins=24)
        plt.plot(rahist, color='blue', linewidth=1)
        plt.xlabel('LST (hours)')
        plt.ylabel('Number of exposures')
        return
