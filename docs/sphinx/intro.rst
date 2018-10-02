
.. _intro:

Introduction to robostrategy
============================

Steps to run strategy
=====================

 * rs_fields -p [plan]
 * rs_slots -p [plan] -o [observatory]
 * rs_targets_ggsp -p [plan] -o [observatory]
 * rs_cadences_ggsp -p [plan] -o [observatory]
 * rs_assign_ggsp -p [plan] -o [observatory]
 * rs_field_slots -p [plan] -o [observatory]
 * rs_allocate -p [plan] -o [observatory]
 * rs_assign_final -p [plan] -o [observatory]
 * rs_assignments -p [plan] -o [observatory]
 * rs_allocate_plot -p [plan] -o [observatory]
 * rs_completeness -p [plan] -o [observatory]
 * rs_completeness_plot -p [plan] -o [observatory]

Things to do
============

 * Fix radec2xy (at least to include LCO)
