
.. _intro:

Introduction to robostrategy
============================

Steps to run strategy
=====================

 * rs_fields -p [plan]
 * rs_slots -p [plan] -o [observatory]
 * rs_targets_straw -p [plan] -o [observatory]
 * rs_cadences_straw -p [plan] -o [observatory]
 * rs_field_targets -p [plan] -o [observatory]
 * rs_field_cadences_straw -p [plan] -o [observatory]
 * rs_field_cadences_plot -p [plan] -o [observatory]
 * rs_assign -p [plan] -o [observatory]
 * rs_field_slots -p [plan] -o [observatory]
 * rs_allocate -p [plan] -o [observatory]
 * rs_assign_final -p [plan] -o [observatory]
 * rs_assignments -p [plan] -o [observatory]
 * rs_allocate_plot -p [plan] -o [observatory]
 * rs_completeness -p [plan] -o [observatory]
 * rs_completeness_plot -p [plan] -o [observatory]

Things to do
============

 * Document all pieces
 * Fix radec2xy (at least to include LCO)
 * Set RM field locations
 * Set RV field locations, and replace RV targets (remove hack in
	 rs_assign)
 * How to make sure targets with field cadence are ensured
 * Mixed BOSS and APOGEE cadence
 * Code to produce report
 * Testbed
