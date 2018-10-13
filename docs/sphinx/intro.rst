
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
 * rs_html -p [plan] -o [observatory]

Things to do
============

 * Make plots
 * Set RM field locations
 * Set RV field locations, and replace RV targets (remove hack in
	 rs_assign)
 * airmass cost
 * Fix radec2xy for 0/360 and poles issues
 * How to make sure targets with field cadence are ensured
 * Fix hex grid pattern
 * Handle assignments in overlapping fields 
 * Mixed BOSS and APOGEE cadence
 * Fix radec2xy fully
 * Testbed
 * Incorporate kaiju constraints
 * Combine LCO/APO constraints into single optimization
 * Put on constraints of number of certain targets to achieve.
