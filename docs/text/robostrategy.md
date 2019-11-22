1. Software and database dependencies

Using robostrategy at Utah requires software to be installed and
access to targeting databases. The databases are accessed near the
beginning of the pipeline (in rs_targets_extract and
rs_cadences_extract) to extract information, but are not needed after
that.

The software used is:

 * robostrategy
 * kaiju
 * roboscheduler
 * sdssdb

and all the code is written against python/3.6.3

To install the software TODO

The db accessed is TODO

2. Overview of pipeline

The settings for the code for a given "plan" are given in a
configuration file, whose options we will describe below:

 $ROBOSTRATEGY_DIR/etc/robostrategy-[plan].cfg

robostrategy outputs files to a directory defined by the environmental
variable ROBOSTRATEGY_DATA:

 $ROBOSTRATEGY_DATA/allocations/[plan]

 * rs_fields: creates the list of fields for APO and LCO to consider.
    rsFields-[plan]-[apo|lco].fits
 * rs_slots: evaluates the schedule and determines dark and bright time per LST
    rsSlots-[plan]-[apo|lco].fits
 * rs_targets_extract: pulls targets out of database (munges some information)
    rsTargets-[plan]-[apo|lco].fits
 * rs_cadences_extract: pulls cadences out of database (munges some information)
    rsCadences-[plan]-[apo|lco].fits
    rsCadences-[plan]-[apo|lco].fits.pkl
 * rs_field_targets: separates targets into fields (for ease of access later)
    targets/rsFieldTargets-[plan]-[apo|lco]-[fieldid].fits
 * rs_field_count: count up different types of targets in each field
    rsFieldCounts-[plan]-[apo|lco].fits
 * rs_field_cadences: decides which cadences should be used for each field
    rsFieldCadences-[plan]-[apo|lco].fits
 * rs_assign: performs assignment for each cadence choice for each field
    rsOptions-[plan]-[apo|lco].fits
 * rs_field_slots: TODO
    rsFieldSlots-[plan]-[apo|lco].fits
 * rs_allocate: decide which cadence to use for each field
    rsAllocation-[plan]-[apo|lco].fits
 * rs_assign_final: perform assignment for the chosen cadence for each field
    TODO
 * rs_completeness: determine completeness of targets and as function of position
 * rs_field_cadences_plot: make QA plot of field cadence choices
 * rs_allocate_plot: make QA plot of LST allocation and field cadence allocation
 * rs_completeness_plot: make QA plots of completeness for cadence and programs
 * rs_html: create HTML file for results


Glossary
--------

* Catalogs: refers to the fundamental catalogs from which the list of
  targets is derived (e.g. Gaia, 2MASS, TESS Input Catalog, etc.).

* Targeting: refers to the process of identifying a catalog object as
  a potential target for spectroscopy. Results in a global list of
  these targets in the database. This does not guarantee a fiber is
  assigned.

* Target: An astronomical object that has been identified in targeting
  to potentially receive a fiber.

* Field: refers to an RA/Dec center associated with a pointing of one
  of the telescopes.

* Design: refers to the design of an observation of the field, which
  consists of which fibers are assigned to which targets.

* Configuration: refers to the conditions of an actual exposure of
  a field, and the specific assignments of fibers to targets.

Standards
---------

 * All target coordinates are given in columns named "ra", "dec",
   which are 64-bit precision and given in J2000 deg, with an "epoch"
   specified in decimal years, and with proper motions called "pmra"
   and "pmdec" in mas / year.

 * All field or design center coordinates are given in columns  named
   "racen", "deccen", which are 64-bit precision and given in J2000
   deg.
 
