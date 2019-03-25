
.. _intro:

Introduction to robostrategy
============================

Steps to run strategy
=====================

 * rs_fields_straw -p [plan]
 * rs_slots -p [plan] -o [observatory]
 * rs_targets_extract -p [plan] -o [observatory]
 * rs_cadences_extract -p [plan] -o [observatory]
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

 * Add sky and standards at some level
 * WD and 100 pc cadence changes
 * Check RV cadence consistency
 * How to make sure targets with field cadence are ensured
 * Fix hex grid pattern
 * twilight time as high lunation
 * Handle assignments in overlapping fields 
 * Mixed BOSS and APOGEE cadence
 * Incorporate kaiju constraints and real radec2xy
 * Combine LCO/APO constraints into single optimization
 * Put on constraints of number of certain targets to achieve.
 * Testbed

Minor things:

 * Plot desired vs RA
 * Add something to rsAllocation that is just 0 or 1, the field is observed.
 * Use targetid not pk for checking completeness
 * Move all stuff to single directory
 * "instrument" in cadence is meaningless for field cadence; what to
   do about that?

Need from teams:

 * In general, review the cadences
 * In general, review the list of targets
 * New RV targets
 * TESS RGB targets?
 * TESS OBAF targets?
 * TESS Planet targets?

