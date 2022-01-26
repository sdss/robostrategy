#!/usr/bin/env python
# -*- coding:utf-8 -*-
# JKC Note: based on spare_fiber_skeleton, but fleshed out

import numpy as np
from robostrategy.field import Field
import roboscheduler.cadence
from robostrategy.field import targets_dtype as targets_dtype
import pdb


# Establish access to the CadenceList singleton
clist = roboscheduler.cadence.CadenceList()


class extra_Field(Field):  #inherit all Field-defined stuff.

    def _redefine_assignments(self):
        # Expand dtype of assignments to include an 'extra' column

        new_dtype = np.dtype(self.assignments_dtype.descr + [('extra',np.int32)])
        new_assignments = np.zeros(len(self.assignments),dtype=new_dtype)
        for name in self.assignments_dtype.names:
            new_assignments[name] = self.assignments[name]

        self.assignments_dtype = new_dtype
        self.assignments = new_assignments

        return

    def assign_spares(self):
        ''' This is the main code that defines all the reassignment that will happen
        in a certain order'''

        # First, redefine assignments to include 'extra'
        self._redefine_assignments()

        # Make the assignments.
        # Return value is either True if any extra assignments occurred or False if not
        extra_dark = self.assign_dark_extra()
        extra_rv = self.assign_rv_extra()
        extra_partial = self.assign_partial()
        extra_bright = self.assign_bright_extra()

        any_success = extra_dark or extra_rv or extra_partial or extra_bright
        return any_success

    def assign_extra(self, rsids=None, max_extra=1, nexps=1, skip_assigned_epochs=True):
        '''
        This is modeled a bit after assign_epochs(). It is a generic code for doing the
        assignments once a set of rsids and maximum requested extras have been identified.

        Parameters:
        ----------
        rsids : ndarray of np.int64
            rsids of targets to assign

        max_extra: np.int
            maximum number of extra epochs to assign (default 1)

        nexps: np.int
            number of exposures for each extra epoch (default 1)

        skip_assigned_epochs: bool
            if the target is already assigned to any exposure in an epoch
            do not assign to another open exposure (default True)


        Returns:
        -------
        nsuccess: ndarray of type np.int
            number of extra epochs successfully assigned for each input rsid


        '''
        nsuccess = np.zeros(len(rsids), dtype=np.int) #count how many extra epochs assigned

        first = True # Only need the first available robot at each epoch?
        nexps_per_epoch = np.full(self.field_cadence.nepochs,nexps)

        for idx,rsid in enumerate(rsids):
            free = self.available_epochs(rsid=rsid, first=first, nexps=nexps_per_epoch)
            # Assign up to max_extra epochs. availableRobotIds is a list of lists
            # Outer list length is field nepoch? Inner list is len n robots (1 if first = True)
            n_assign = 0

            iassigned = self.assignments['equivRobotID'][self.rsid2indx[rsid]] >= 0

            # If skip_assigned_epochs, then if any exposure in an epoch is assinged,
            # mark them all as previously assigned. Otherwise, you are assigning
            # extra exposures, not epochs
            u_nexp = np.unique(self.field_cadence.nexp)

            if len(u_nexp) != 1:
                raise ValueError("More than one nexp found for this cadence. Need to recode logic")

            if skip_assigned_epochs and u_nexp[0] > 1:
                for iep in np.arange(0,self.field_cadence.nepochs):
                    inexp = self.field_cadence.nexp[iep]
                    if np.any(iassigned[iep*u_nexp[0]: iep*u_nexp[0] + u_nexp[0]]):
                        iassigned[iep*u_nexp[0] : iep*u_nexp[0] + u_nexp[0]] = True

            for iepoch, per_epoch_available in enumerate(free['availableRobotIDs']):
                if iassigned[iepoch]:
                    continue # target already assigned

                if len(per_epoch_available) > 0:
                    first_free_robot = per_epoch_available[0]
                    is_assign = self.assign_robot_epoch(rsid=rsid, robotID=first_free_robot, epoch=iepoch,
                                                        status=free['statuses'][iepoch][0], nexp=nexps)
                    if is_assign:
                       n_assign += 1
                if n_assign >= max_extra: #stop when you have hit maximum extra
                    break


            nsuccess[idx] = n_assign

            self.assignments['extra'][self.rsid2indx[rsid]] = n_assign

        self.decollide_unassigned()

        return(nsuccess)


    def assign_dark_extra(self,make_report=False):
        '''
        Code for assigning extra MWM dark time targets. Does nothing if the field
        is a bright-only field. Otherwise, identify previously "satisfied" objects
        for WDs, SNC (100/250pc boss), and BOSS CB targets (300pc, gaiagalex,
        cvcandidates). And check for extra epochs.

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''

        max_extra = 99 #get as many as you can, but for testing start with 1
        any_extra = False # initialize

        #Check whether field contains any dark exposures
        if not clist.cadence_consistency('dark_1x1', self.field_cadence.name, return_solutions=False):
            return any_extra

        # Find gotten WDs and try to get extra epochs
        iextra = np.where((self.targets['carton'] == 'mwm_wd_core') &
                          (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=max_extra)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('White Dwarfs')
                print('------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))


        # Next find gotten SNC and try to get extra epochs
        iextra = np.where(((self.targets['carton'] == 'mwm_snc_100pc_boss') |
                           (self.targets['carton'] == 'mwm_snc_250pc_boss')) &
                           (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess =  self.assign_extra(rsids=extra_rsids, max_extra=max_extra)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('Solar Neighborhood')
                print('------------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))

        # Next find compact binaries and get extra epochs
        iextra = np.where(((self.targets['carton'] == 'mwm_cb_300pc_boss') |
                           (self.targets['carton'] == 'mwm_cb_gaiagalex_boss') |
                           (self.targets['carton'] == 'mwm_cb_cvcandidates_boss')) &
                           (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=max_extra)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('Compact Binaries')
                print('----------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))
        return any_extra

    def assign_rv_extra(self, make_report=False):
        '''
        Code for assigning extra epochs to RV targets. The more the merrier for these stars!

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened

        '''

        any_extra = False

        # Find gotten RVs and see try to get extra epochs - take any in the RV
        # program that was satisfied. Additionally, take any mwm_rv_long that
        # is UNSATISFIED and do those first
        iextra1 = np.where((self.targets['carton'] == 'mwm_rv_long_fps') &
                           (self.assignments['satisfied'] == 0))[0]
        iextra2 = np.where((self.targets['program'] == 'mwm_rv') &
                           (self.assignments['satisfied'] > 0))[0]
        iextra = np.append(iextra1,iextra2)

        if len(iextra) > 0:
            # For RV's, group by cadence so you can determine how many exposures
            ucad = np.unique(self.targets['cadence'][iextra])

            for icad in ucad:
                subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                nexps = clist.cadences[icad].nexp[0]
                extra_rsids = self.targets['rsid'][iextra[subset]]
                nsuccess = self.assign_extra(rsids=extra_rsids, max_extra = 99, nexps=nexps)
                if len(nsuccess[nsuccess > 0]) > 0:
                   any_extra = True

                if make_report:
                    print('Radial Velocity')
                    print('---------------')
                    print('Number attempted: {}'.format(len(iextra)))
                    print('Number successful: ')
                    uextra,ctextra = np.unique(nsuccess, return_counts=True)
                    for iex,ict in zip(uextra,ctextra):
                        print(f'   {ict} stars - {iex} extra epochs')
        return any_extra

    def assign_partial(self, make_report=False):
        '''
        Code for assigning MWM targets with secondary science goals achievable
        if < than the full cadence is obtained. Just YSOs for now

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''

        any_extra = False

        # Find NOT-gotten YSOs and try to get some epochs
        iextra = np.where((self.targets['program'] == 'mwm_yso') &
                          (self.assignments['satisfied'] == 0))[0]

        if len(iextra) > 0:
            # For YSO's, group by cadence so you can determine how many exposures
            ucad = np.unique(self.targets['cadence'][iextra])

            for icad in ucad:
                subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                nexps = clist.cadences[icad].nexp[0] #This is an array of length nepoch, but assign_extra expects int
                nepochs = clist.cadences[icad].nepochs
                extra_rsids = self.targets['rsid'][iextra[subset]]
                nsuccess = self.assign_extra(rsids=extra_rsids, max_extra = nepochs, nexps=nexps)
                if len(nsuccess[nsuccess > 0]) > 0:
                    any_extra = True
                uextra,ctextra = np.unique(nsuccess, return_counts=True)

                if make_report:
                    print(f'\nPartial Epochs (YSOs): {icad}')
                    print('--------------------')
                    print('Number attempted: {}'.format(len(subset)))
                    print('Number successful: ')
                    for iex,ict in zip(uextra,ctextra):
                        print(f'   {ict} stars - {iex} extra epochs')
                    print(extra_rsids)
                    print(nsuccess)
        return any_extra

    def assign_bright_extra(self, make_report=False):
        '''
        Code for assigning extra epochs to MWM bright time targets. Identify previously
        "satisfied" objects in the OB carton.

        Then assign first partial then extra EXPOSURES for BHM

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''
        max_extra = 99 #get as many as you can, but for testing start with 1
        any_extra = False # initialize

        # Find gotten OB stars and try to get extra epochs. Do whole program
        iextra = np.where((self.targets['program'] == 'mwm_ob') &
                          (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=max_extra)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('OB stars')
                print('------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))

        # Find BRIGHT unsatisfied targets in BHM

        return any_extra
