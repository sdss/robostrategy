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

Batch PBS and scripts

2. Overview of pipeline

The settings for the code for a given "plan" are given in a
configuration file, whose options we will describe below:

 $ROBOSTRATEGY_DIR/etc/robostrategy-[plan].cfg

robostrategy outputs files to a directory defined by the environmental
variable ROBOSTRATEGY_DATA:

 $ROBOSTRATEGY_DATA/allocations/[plan]

The code proceeds first by creating the field list, determining how
much time is available in the schedule, and by extracting the targets
and cadences from teh database. Then it tries all of the relevant
cadences for each field. Then it decides on which cadence to use for
which field by maximizing the total "value" of the targets thereby
assigned. Finally, it performs the final assignment for the cadences
chosen, and then makes a number of QA calculations.

The code proceeds through a series of Python scripts. I run these in a
batch PBS script submitted to the Utah cluster.

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
    targets/rsFieldAssignments-[plan]-[apo|lco]-[fieldid].fits
 * rs_assigments: gather all field assignments into a single file
    rsAssignments-[plan]-[apo-lco].fits
 * rs_completeness: determine completeness of targets and as function of position
    rsCompleteness-[plan]-[apo|lco].fits
 * rs_field_cadences_plot: make QA plot of field cadence choices
 * rs_allocate_plot: make QA plot of LST allocation and field cadence allocation
 * rs_completeness_plot: make QA plots of completeness for cadence and programs
 * rs_html: create HTML file for results

Each of these scripts take the two required arguments "-o
<observatory>" (where observatory is 'apo' or 'lco') and "-p <plan>"
(for the plan name to use).

3. Details on Each Python Script

3.a. rs_fields 

 Description:
   Creates list of fields, including all-sky fields, RM fields, RV fields
 Calling Sequence:
   rs_fields -p <plan>
 Inputs: 
   $ROBOSTRATEGY_DIR/data/north-fields-RM-v2.fits
   $ROBOSTRATEGY_DIR/data/south-fields-RM-v2.fits
   $ROBOSTRATEGY_DIR/data/MWM-RV-FieldCenters-v05.fits
 Outputs:
   $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-apo.fits
   $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-lco.fits
 Configation options:
   Fields.NSDiv : Declination division between APO & LCO (degrees)
   Fields.lcoextend_gp : if set, extend LCO coverage in GP northward
   Fields.lcoextend_ngc : if set, extend LCO coverage in NGC northward

3.b. rs_slots

 Description:
   Uses desired schedule to determine available dark & bright time vs. LST
 Calling Sequence:
  rs_slots -p <plan> -o <observatory>
 Inputs:
  $ROBOSCHEDULER_DIR/data/master_schedule_[observatory]_[Schedule].par
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsSlots-[plan]-[observatory].fits
 Configuration options:
  Allocation.Schedule : name of schedule to use

3.c. rs_targets_extract

 Description:
   Extracts targets from database, and also inserts fake sky & standard targets.
   Rescales values of targets if so desired.
 Calling Sequence:
  rs_targets_extract -p <plan> -o <observatory>
 Inputs:
  targetdb
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargets-[plan]-[observatory].fits
 Configuration options:
  None
 
3.d. rs_cadences_extract

 Description:
   Extracts cadences from database, with some adjustments applied.
   Most common is "NoDelta"
   But it also has a hard-coded adjustment that makes RV more lenient.
   It creates a .pkl file that has some predetermined cadence consistencies
 Calling Sequence:
  rs_cadences_extract -p <plan> -o <observatory>
 Inputs:
  targetdb
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits.pkl
 Configuration options:
  Cadences.[cadence] : adjustments to apply, including
     NoDelta - do not impose any requirements on when exposures are taken

3.e rs_field_targets

 Description:
  Pulls out targets relevant to each field into separate files.
  Also determines which targets are covered by fields.
 Calling Sequence:
  rs_field_targets -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargets-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldTargets-[plan]-[observatory]-[fieldid].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargetsCovered-[plan]-[observatory].fits
 Configuration options:
  None

3.f. rs_field_count

 Description:
  Using rsFieldTargets files, counts number of targets within field
   of each cadence and type.
 Calling Sequence:
  rs_field_count -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargets-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldTargets-[plan]-[observatory]-[fieldid].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFieldCount-[plan]-[observatory].fits
 Configuration options:
  None

3.g. rs_field_cadences

 Description:
  Determine which cadences should be tried for each field, based on 
   targets within the field (or based on field type).
 Calling Sequence:
  rs_field_cadences -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldTargets-[plan]-[observatory]-[fieldid].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFieldCadences-[plan]-[observatory].fits
 Configuration options:
  Fields.[type] : list of cadences to try from type (or "FromTargets")
  CadencesFromTargets.[target_cadence] : field cadence list to try (if nothing listed, use target_cadence as a field cadence to try

3.h. rs_assign

 Description:
  Assign targets for each cadence and field combination. Outputs
  results on how many targets and what total value results from each 
  cadence. Does NOT account for calibration targets or collision 
  constraints.
 Calling Sequence:
  rs_assign -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFieldCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargets-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFieldCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldTargets-[plan]-[observatory]-[fieldid].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsOptions-[plan]-[observatory].fits
 Configuration options:
  Assignment.fgot_minimum : for single-n cadences, only keep choices that
    get at least this fraction of Galactic Genesis targets
  Assignment.fgot_maximum : for single-n cadences, only keep choices that
    get no more than this fraction of Galactic Genesis targets

3.i. rs_field_slots

 Description:
  Determine which LST and sky brightness hours can be used for each 
  field-cadence combination in rsOptions
 Calling Sequence:
  rs_field_slots -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsSlots-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsOptions-[plan]-[observatory].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFieldSlots-[plan]-[observatory].fits
 Configuration options:
  Allocation.BrightOnlyBright : set if bright cadences only can get bright time

3.j. rs_allocate

 Description:
  Decide which cadences to use for each field, by maximizing value 
  under resource (and some other) constraints.
 Calling Sequence:
  rs_allocate -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsSlots-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFields-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsOptions-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsFieldSlots-[plan]-[observatory].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsAllocation-[plan]-[observatory].fits
 Configuration options:
  Allocation.AllFields : list of field types which should be guaranteed
  Allocation.DarkPrefer : factor to overvalue a dark observation [not recommended]
  Allocation.Cost : model to use for cost (A, B, C, or D; D is the right one)
  AllocationMinimumTargetsAtAPO.[program] : minimum number of targets for program at APO
  AllocationMinimumTargetsAtAPO.[program] : minimum number of targets for program at LCO
  ValueRescale.[cadence] : rescaling to apply to value of target cadence

3.k. rs_assign_final

 Description:
  Perform assignment of targets using chosen cadence, accounting 
  for calibration and collision constraints.
 Calling Sequence:
  rs_assign_final -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsAllocation-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldTargets-[plan]-[observatory]-[fieldid].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldAssignment-[plan]-[observatory]-[fieldid].fits
 Configuration options:
  None

3.l. rs_assignments

 Description:
  Gather the assignments into a single file of target-exposures
 Calling Sequence:
  rs_assignments -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCadences-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsAllocation-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldAssignments-[plan]-[observatory]-[fieldid].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsAssignments-[plan]-[observatory].fits
 Configuration options:
  None

3.m. rs_completeness

 Description:
  Evaluate completeness of targets individually, and as a function of healpix pixel
 Calling Sequence:
  rs_completeness -p <plan> -o <observatory>
 Inputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargets-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsTargetsCovered-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsAssignments-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsAllocation-[plan]-[observatory].fits
  $ROBOSTRATEGY_DATA/allocations/[plan]/targets/rsFieldTargets-[plan]-[observatory]-[fieldid].fits
 Outputs:
  $ROBOSTRATEGY_DATA/allocations/[plan]/rsCompleteness-[plan]-[observatory]-[fieldid].fits
 Configuration options:
  None

Outline of process
------------------

Describe field choices

Describe slots

Describe cadence choices

Describe allocation decisions

Describe assignment process

Looking at the results
----------------------

Example cfg file explained

rsAllocation file (and plots)

rsFieldAssignments file

rsCompleteness file (and plots)

Important Missing Pieces
------------------------

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
 
