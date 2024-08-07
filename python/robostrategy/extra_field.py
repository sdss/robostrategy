#!/usr/bin/env python
# -*- coding:utf-8 -*-
# JKC Note: based on spare_fiber_skeleton, but fleshed out

import numpy as np
from robostrategy.field import Field
import roboscheduler.cadence

# Establish access to the CadenceList singleton
clist = roboscheduler.cadence.CadenceList(skybrightness_only=True)


class extra_Field(Field):  #inherit all Field-defined stuff.

    def assign_spares(self, stage='reassign',verbose=False):
        ''' This is the main code that defines all the reassignment that will happen
        in a certain order'''

        self.set_stage(stage=stage)

        # Make the assignments.
        # Return value is either True if any extra assignments occurred or False if not
        extra_dark = self.assign_dark_extra(make_report=verbose)
        extra_rv = self.assign_rv_extra(make_report=verbose)
        extra_partial = self.assign_partial(make_report=verbose)
        extra_bhm1 = self.assign_bhm_partial(make_report=verbose)
        extra_bright = self.assign_bright_extra(make_report=verbose)
        extra_bhm2 = self.assign_bhm_extra(make_report=verbose)

        self._set_satisfied()
        self._set_satisfied(science=True)
        self._set_count(reset_equiv=False)
        self.set_stage(stage=None)

        any_success = extra_dark or extra_rv or extra_partial or extra_bright or extra_bhm1 or extra_bhm2
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
        nsuccess = np.zeros(len(rsids), dtype=np.int)  # count how many extra epochs assigned

        first = True  # Only need the first available robot at each epoch?
        nexps_per_epoch = np.full(self.field_cadence.nepochs,nexps)

        for idx,rsid in enumerate(rsids):
            free = self.available_epochs(rsid=rsid, first=first, nexps=nexps_per_epoch)
            # Assign up to max_extra epochs. availableRobotIds is a list of lists
            # Outer list length is field nepoch? Inner list is len n robots (1 if first = True)
            n_assign = 0

            # If skip_assigned_epochs, then if any exposure in an epoch is assinged,
            # mark them all as previously assigned. Otherwise, you are assigning
            # extra exposures, not epochs
            iassigned_epoch = np.full(self.field_cadence.nepochs, False)
            if skip_assigned_epochs:
                for iexp, rid in enumerate(self.assignments['equivRobotID'][self.rsid2indx[rsid]]):
                    if rid >= 0:
                        iassigned_epoch[self.field_cadence.epochs[iexp]] = True


            for iepoch, per_epoch_available in enumerate(free['availableRobotIDs']):
                if iassigned_epoch[iepoch]:
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

            self.assignments['extra'][self.rsid2indx[rsid]] += n_assign

        self.decollide_unassigned()

        return(nsuccess)

    def assign_extra_exps(self, rsids=None, max_extra=1):
        '''
        This is a generic code for assigning exposures outside of the nominal target cadence.
        This is modeled a bit after assign_cadences().
        Parameters:
        ----------
        rsids : ndarray of np.int64
            rsids of targets to assign
        max_extra: np.int
            maximum number of extra exposures to assign (default 1)
        Returns:
        -------
        nsuccess: ndarray of type np.int
            number of exposures successfully assigned for each input rsid
        '''
        nsuccess = np.zeros(len(rsids), dtype=np.int) #count how many extra exposures assigned

        first = True # Only return first available robot at each epoch
                     # Simplest to code but may be limiting for cases when max_extra > 1
                     # Other available robots could have more exposures available
        iexps = np.arange(len(self.assignments['robotID'][0]))
        for idx,rsid in enumerate(rsids):
            free = self.available_epochs(rsid=rsid, first=first) # By default requests 1 exposures

            n_assign = 0
            not_assigned = iexps[(self.assignments['equivRobotID'][self.rsid2indx[rsid]] < 0)]

            n_avail = len(not_assigned)
            if n_avail <= max_extra:
                done = self.assign_exposures(rsid=rsid, iexps=not_assigned)
                n_assign = n_assign + done.sum()
            else:
                done = 0
                for one_exp in not_assigned:
                    one_done = self.assign_exposures(rsid=rsid, iexps=np.array([one_exp]))
                    done = done + one_done
                    if done >= max_extra:
                        break
                n_assign = n_assign + done

            nsuccess[idx] = n_assign

            self.assignments['extra'][self.rsid2indx[rsid]] = self.assignments['extra'][self.rsid2indx[rsid]] + n_assign

        self.decollide_unassigned()

        return(nsuccess)


    def assign_dark_extra(self,make_report=False):
        '''
        Code for assigning extra MWM dark time targets. Does nothing if the field
        is a bright-only field. Otherwise, identify previously "satisfied" objects
        for WDs, SNC (100/250pc boss), and BOSS CB targets (300pc, gaiagalex,
        cvcandidates). After eta-2-candidate-0, this switched from "extra epochs"
        to "extra exposures"

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''

        max_extra = 99 #get as many as you can, but for testing start with 1
        any_extra = False # initialize

        #Check whether field contains any dark exposures
        if not clist.cadence_consistency('_field_dark_single_1x1', self.field_cadence.name, return_solutions=False):
            return any_extra

        # Find gotten WDs and try to get extra epochs
        iextra = np.where((self.targets['carton'] == 'mwm_wd_pwd_boss') &
                          (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess = self.assign_extra_exps(rsids=extra_rsids, max_extra=max_extra)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('White Dwarfs')
                print('------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))


        # Next find gotten SNC and try to get extra epochs
        iextra = np.where(((self.targets['carton'] == 'mwm_snc_100pc_boss') |
                           (self.targets['carton'] == 'mwm_snc_ext_main_boss')) &
                           (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess =  self.assign_extra_exps(rsids=extra_rsids, max_extra=max_extra)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('Solar Neighborhood')
                print('------------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))

        # Next find compact binaries and get extra epochs
        iextra = np.where(((self.targets['carton'] == 'mwm_cb_galex_vol_boss') |
                           (self.targets['carton'] == 'mwm_cb_galex_mag_boss') |
                           (self.targets['carton'] == 'mwm_cb_cvcandidates_boss')) &
                           (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess = self.assign_extra_exps(rsids=extra_rsids, max_extra=max_extra)
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
        Code for assigning extra epochs to RV targets.

        Parameters
        ----------
        make_report: bool
            if True print out a report of what happened

        '''

        any_extra = False
        max_extra = 18

        # Find gotten RVs and see try to get extra epochs - take any in the RV
        # bin cartons that were satisfied. Additionally, take any mwm_rv_long or
        # manual_mwm_validation_rv_apogee targets that
        # are UNSATISFIED and do those first
        iextra1 = np.where(((self.targets['carton'] == 'mwm_bin_rv_long_apogee') | 
                            (self.targets['carton'] == 'manual_mwm_validation_rv_apogee')) &
                           (self.assignments['satisfied'] == 0))[0]
        isrv_carton = np.array(['_rv_' in carton for carton in self.targets['carton']],
                               dtype=bool)
        iextra2 = np.where((isrv_carton) & (self.assignments['satisfied'] > 0))[0]
        iextra = np.append(iextra1,iextra2)

        if len(iextra) > 0:
            # For RV's, group by cadence so you can determine how many exposures
            ucad = np.unique(self.targets['cadence'][iextra])

            for icad in ucad:
                subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                nexps = clist.cadences[icad].nexp[0]
                extra_rsids = self.targets['rsid'][iextra[subset]]
                nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=max_extra, nexps=nexps)
                if len(nsuccess[nsuccess > 0]) > 0:
                   any_extra = True

                if make_report:
                    print('Radial Velocity')
                    print('---------------')
                    print('Number attempted: {}'.format(len(subset)))
                    print('Number successful: ')
                    uextra,ctextra = np.unique(nsuccess, return_counts=True)
                    for iex,ict in zip(uextra,ctextra):
                        print(f'   {ict} stars - {iex} extra epochs')
        return any_extra

    def assign_partial(self, make_report=False):
        '''
        Code for assigning MWM targets with secondary science goals achievable
        if < than the full cadence is obtained: mwm_tess_2min, as a stop gap
        for fixing "flexible cadence" implementation; then YSOs.

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''

        any_extra = False

        # Find NOT-gotten mwm_tess_planet and swap cadence from 1xN to Nx1
        iextra = np.where((self.targets['carton'] == 'mwm_tess_2min_apogee') &
                          (self.assignments['satisfied'] == 0))[0]

        if len(iextra) > 0:
            # Group by cadence to max out at requested N
            ucad = np.unique(self.targets['cadence'][iextra])

            for icad in ucad:
                subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                nexps = clist.cadences[icad].nexp[0]
                nepochs = clist.cadences[icad].nepochs

                # for this target class, cadences are currently in 1xN format but
                # may change back to Nx1. For assign_spares, want requested exposures
                # = 1 and requsted epochs = N
                nexps_need = 1
               
                if nexps > 1:
                    if nepochs > 1:
                        raise ValueError("Did not expect mwm_tess_planet cadence to have both nexp and nepoch > 1")
                    nepochs_need = nexps
                else:
                    nepochs_need = nepochs
         

                extra_rsids = self.targets['rsid'][iextra[subset]]
                nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=nepochs_need,
                                             nexps=nexps_need)
                if len(nsuccess[nsuccess > 0]) > 0:
                    any_extra = True
                uextra,ctextra = np.unique(nsuccess, return_counts=True)

                if make_report:
                    print(f'\nPartial Epochs (TESS Planet): {icad}')
                    print('--------------------')
                    print('Number attempted: {}'.format(len(subset)))
                    print('Number successful: ')
                    for iex,ict in zip(uextra,ctextra):
                        print(f'   {ict} stars - {iex} extra epochs')
                    print(extra_rsids)
                    print(nsuccess)


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
                nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=nepochs, nexps=nexps)
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

        # Find NOT-gotten OBA core stars and try to get some epochs
        # Note: Do not need to do this on mwm_ob_core_boss_single, which are same targets
        iextra = np.where(((self.targets['carton'] == 'mwm_ob_core_boss') |  
                           (self.targets['carton'] == 'mwm_astar_core_boss')) &
                          (self.assignments['satisfied'] == 0))[0]

        if len(iextra) > 0:
            ucad = np.unique(self.targets['cadence'][iextra])

            for icad in ucad:
                subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                nexps = clist.cadences[icad].nexp[0] #This is an array of length nepoch, but assign_extra expects int
                nepochs = clist.cadences[icad].nepochs
                extra_rsids = self.targets['rsid'][iextra[subset]]
                nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=nepochs, nexps=nexps)
                if len(nsuccess[nsuccess > 0]) > 0:
                    any_extra = True
                uextra,ctextra = np.unique(nsuccess, return_counts=True)

                if make_report:
                    print(f'\nPartial Epochs (OB Stars): {icad}')
                    print('--------------------')
                    print('Number attempted: {}'.format(len(subset)))
                    print('Number successful: ')
                    for iex,ict in zip(uextra,ctextra):
                        print(f'   {ict} stars - {iex} extra epochs')
                    print(extra_rsids)
                    print(nsuccess)

        return any_extra

    def assign_bright_extra(self, make_report=False):
        '''Code for assigning extra epochs to MWM bright time targets:
         * Previously "satisfied" targets in TESS Planet
         * Previous "partial" targets in TESS planet (b/c they were not allowed
           over N requested at the partial completion stage)
         * Previously "satisfied" objects in the OB carton.

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''
        max_extra_planet = 12
        max_extra_ob = 3
        any_extra = False # initialize

        # Find TESS planet targets that are 'satisfied' or that previously got
        # extra, since extra was originally capped at N epochs but at this
        # later stage they are now eligible for extra epochs
        iextra = np.where((self.targets['carton'] == 'mwm_tess_2min_apogee') &
                         ((self.assignments['satisfied'] > 0) | (self.assignments['extra'] > 0)))[0]

        if len(iextra) > 0:
            extra_rsids = self.targets['rsid'][iextra]
            nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=max_extra_planet)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('\nExtra Epochs (TESS Planet stars)')
                print('------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))

        # Find gotten OBA stars and try to get extra epochs. 
        iextra = np.where(((self.targets['carton'] == 'mwm_ob_core_boss') |
                           (self.targets['carton'] == 'mwm_astar_core_boss')) &
                          (self.assignments['satisfied'] > 0))[0]
        if len(iextra) > 0:
            extra_rsids = self.targets["rsid"][iextra]
            nsuccess = self.assign_extra(rsids=extra_rsids, max_extra=max_extra_ob)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra = True
            if make_report:
                print('\nExtra Epochs (OB stars)')
                print('------------')
                print('Number attempted: {}'.format(len(iextra)))
                print('Number successful: {}\n'.format(len(nsuccess[nsuccess > 0])))


        return any_extra

    def assign_bhm_partial(self,make_report=False):
        '''
        Code for assigning partial exposures to BHM targets. For partial
        completion, original cadence details are ignored and just to get the total
        requested nxm request. 

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''

        any_extra = False # initialize

        # Find all eligible targets for partial cadence completion
        iextra1 = np.where((self.targets['program'] == 'bhm_spiders') &
                            (self.assignments['satisfied'] == 0))[0]
        iextra2 = np.where((self.targets['program'] == 'bhm_csc') &
                            (self.assignments['satisfied'] == 0))[0]
        iextra3 = np.where((self.targets['carton'] == 'bhm_gua_dark') &
                            (self.assignments['satisfied'] == 0))[0]
        iextra4 = np.where((self.targets['carton'] == 'bhm_gua_bright') &
                            (self.assignments['satisfied'] == 0))[0]

        iextra = np.append(iextra1, np.append(iextra2, np.append(iextra3, iextra4)))

        # Is this a dark field?
        is_dark_field = clist.cadence_consistency('_field_dark_single_1x1', self.field_cadence.name,
                                                   return_solutions=False)
        if len(iextra > 0):
            ucad = np.unique(self.targets['cadence'][iextra])

            for icad in ucad:
                if 'dark' in icad and not is_dark_field: #skip dark targets if not in dark field
                    continue

                subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                max_extra = clist.cadences[icad].nexp[0] * clist.cadences[icad].nepochs
                if len(subset) > 1:
                    isort = np.argsort(self.targets['priority'][iextra[subset]])
                    extra_rsids = self.targets['rsid'][iextra[subset[isort]]]
                else:
                    extra_rsids = self.targets['rsid'][iextra[subset]]
                nsuccess = self.assign_extra_exps(rsids=extra_rsids, max_extra=max_extra)
                if len(nsuccess[nsuccess > 0]) > 0:
                    any_extra=True

                if make_report:
                    print(f'\nPartial Exposures for BHM:{icad}')
                    print('--------------------------------')
                    print("Number attempted: {}".format(len(subset)))
                    print('Number successful: ')
                    uextra,ctextra = np.unique(nsuccess, return_counts=True)
                    for iex,ict in zip(uextra,ctextra):
                        print(f'    {ict} stars - {iex} extra exposures')
                    print(extra_rsids)
                    print(nsuccess)


    def assign_bhm_extra(self,make_report=False):
        '''
        Code for assigning extra exposures to BHM targets and magcloud targets

        Parameters:
        ----------
        make_report: bool
            if True print out a report of what happened
        '''

        any_extra = False # initialize
        max_extra_bhm = 4
        max_extra_magcloud = 8

        # Is this a dark field?
        is_dark_field = clist.cadence_consistency('_field_dark_single_1x1', self.field_cadence.name,
                                                   return_solutions=False)


        # Find all eligible targets for extra completion: Those with extra assignments
        # from assign_bhm_partial + previouly satisfied
        iextra1 = np.where((self.targets['program'] == 'bhm_spiders') &
                            (self.assignments['satisfied'] > 0) | (self.assignments['extra'] > 0))[0]
        iextra2 = np.where((self.targets['program'] == 'bhm_csc') &
                            (self.assignments['satisfied'] > 0) | (self.assignments['extra'] > 0))[0]
        iextra3 = np.where((self.targets['carton'] == 'bhm_gua_dark') &
                            (self.assignments['satisfied'] > 0) | (self.assignments['extra'] > 0))[0]
        iextra4 = np.where((self.targets['carton'] == 'bhm_gua_bright') &
                            (self.assignments['satisfied'] > 0) | (self.assignments['extra'] > 0))[0]

        #some "new_got" may now also show up as 'satisfied'
        iextra = np.unique(np.append(iextra1, np.append(iextra2, np.append(iextra3,iextra4))))

        # In this situation, only use the cadence groupings to remove dark cadence targets
        # in bright fields
        if len(iextra > 0):
            ucad = np.unique(self.targets['cadence'][iextra])
            kp = np.full(len(iextra), True)
            for icad in ucad:
                if 'dark' in icad and not is_dark_field: #skip dark targets if not in dark field
                    subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                    kp[subset] = False
            iextra = iextra[kp]
            isort = np.argsort(self.targets['priority'][iextra])
            extra_rsids = self.targets['rsid'][iextra[isort]]
            nsuccess = self.assign_extra_exps(rsids=extra_rsids, max_extra=max_extra_bhm)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra=True

            if make_report:
                print(f'\nExtra Exposures for BHM:')
                print('--------------------------------')
                print("Number attempted: {}".format(len(iextra)))
                print('Number successful: ')
                uextra,ctextra = np.unique(nsuccess, return_counts=True)
                for iex,ict in zip(uextra,ctextra):
                    print(f'    {ict} stars - {iex} extra exposures')
                print(extra_rsids)
                print(nsuccess)

        # repeat for magellanic cloud
        iextra = np.where(((self.targets['program'] == 'mwm_magcloud') |
                           (self.targets['program'] == 'mwm_magcloud_agb') |
                           (self.targets['program'] == 'mwm_magcloud_rgb')) &
                          (self.assignments['satisfied'] > 0))[0]

        if len(iextra > 0):
            ucad = np.unique(self.targets['cadence'][iextra])
            kp = np.full(len(iextra), True)
            for icad in ucad:
                if 'dark' in icad and not is_dark_field: #skip dark targets if not in dark field
                    subset = np.where(self.targets['cadence'][iextra] == icad)[0]
                    kp[subset] = False
            iextra = iextra[kp]
            isort = np.argsort(self.targets['priority'][iextra])
            extra_rsids = self.targets['rsid'][iextra[isort]]
            nsuccess = self.assign_extra_exps(rsids=extra_rsids, max_extra=max_extra_magcloud)
            if len(nsuccess[nsuccess > 0]) > 0:
                any_extra=True

            if make_report:
                print(f'\nExtra Exposures for Mag Cloud:')
                print('--------------------------------')
                print("Number attempted: {}".format(len(iextra)))
                print('Number successful: ')
                uextra,ctextra = np.unique(nsuccess, return_counts=True)
                for iex,ict in zip(uextra,ctextra):
                    print(f'    {ict} stars - {iex} extra exposures')
                print(extra_rsids)
                print(nsuccess)

        return any_extra
