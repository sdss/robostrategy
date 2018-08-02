This document describes a conceptual design for the targeting and
scheduling system for SDSS-V.

Elements of Observing Strategy
------------------------------

1. Targeting

 a. Fundamental catalogs (Gaia, 2MASS, TIC, SDSS imaging, SDSS
 spectroscopy, others) are loaded into catalogdb. catalogdb has
 spatial indexing (q3c). 

 b. Fundamental catalogs are resolved into a single "catalog" table.
 All targets except targets-of-opportunity must exist in this table.
 Each entry receives a unique catalogid. The catalog stores the
 healpix pixel (nside=64, which is a bit less than 1 deg on a side)
 containing the RA/Dec.

 c. "target" is run. It receives input only from catalogdb. The
 operations team provides tools to extract catalog objects and their
 information, and to write results into targetdb. The mapper teams
 provide code that identifies targets using those tools to interact
 with the databases. Skies and standards are included in the list of
 targets.  A notion would be to run this code healpix by healpix,
 since interacting with the database on an object-by-object basis may
 introduce excessive overhead. The target process produces a table in
 targetdb which has:

  * catalogdb ID
  * targeting bit mask
  * value(s)
  * cadence category
  * version

 How we handle versioning is a bit unclear as yet.

 Each cadence category has the following information:

  * nexposures:
	    number of exposures in the field total.
  * delta[]:
	    for each exposure, ideal number of days since last exposure
  * delta_low[]:
	    for each exposure, tolerance on low side of delta
  * delta_high[]:
	    for each exposure, tolerance on high side of delta
  * lunation[]:
	    for each epoch, the maximum moon lunation (lunation is equal to
	    moon illumination, or 0 if moon is below the horizon)

 d. The target table is output to disk as a set of FITS files
 organized by healpix pixel, for archival purposes. 

2. Strategic Design

This occurs prior to the survey, and its purpose is to establish an
overall plan that is workable and will achieve the goals of the
project. We expect these tasks to be iterated with changes in choices
and approaches based on simulations.

 a. A set of fields are defined that cover the area with the
    appropriate density. This will almost certainly be a by-hand task.

 b. "design" is run for each field. For each field, we choose a set of
	   possible cadences, and assign targets to fibers under that
	   cadence. Our first path forward will be as follows: (i) to
	   consider the unassigned targets available to each robot in turn,
	   and pack the epochs of that robot with the targets to maximize
	   total value; (ii) to allow trades at each epoch to free up robots
	   that are then allocated again. At this stage we need to
	   incorporate the robot constraints, which is not solved at this
	   point. Skies and standards are included. A total value for each
	   cadence choice is defined based on the net results.

 c. We then model the survey time availability as a function of
    LST. There is a total number of exposures available at each
    LST. For each LST, a specific set of fields is observable. 
		Each field can be observed under one choice of cadence, which
		costs a certain number of exposures. The result tells us which
		cadences to approach in which field will maximize the value.

 d. We assess the results in terms of the targets achieved and
	  fields observed and adjust values until a satisfactory result
		emerges.

3. Simulation:

 a. Define a strategy for selecting the next design based on its
 observability and its cadence. The strategy may allow a
 periodic rerunning of design based on achieved results.  The strategy
 may have tunable parameters. 

 b. Run simulations under that strategy.

 c. Iterate on the strategy parameters to achieve best results on
 some global metric.

 d. Assess strategy based on individual program outcomes.

 e. Iterate on choices of the global metric in (c), the strategy (a),
 and also the values that go into the strategic design, to best
 achieve outcomes of (d), whatever that means.

4. Survey operations

 a. Prior to the survey, the targets and designs are prepared. This means
    designing and running targeting software across the whole sky, and 
    designing a full set of designs.

 b. Each night, the strategy from (3) is used throughout the night to
    pick the next design to observe. At this point
    targets-of-opportunity may be added. The configuration used for
    each observation is stored and the results of each exposure is
    stored. At this stage we use S/N criteria to decide whether to
		an full exposure is achieved on a design.

 c. Each morning, the fpsdb database is updated with the results of
    the night. The status of each design and overall fiber plan for
    the designs may be updated as well. This redesign may mean, for
    example, that individual fibers that performed either better or
    worse than expected may be reassigned in the future-looking fiber
    plan (expressed in the "fiber" table of fpsdb).

fpsdb should be designed to be lightweight. It would be instantiated
at Utah, LCO, and APO. Targets would be designated for one or other or
both. A daytime process would need to update the Utah version based on
LCO and based on APO, perform redesign, and push back the full updated
database to the observatories. If connectivity was lost, the redesign
operation would be skipped that day and updates would happen later.

Software products
-----------------

Right now everything is being developed in observesim. However,
ultimately we should be refactoring into separate products:

* sdssdb: to create and access fpsdb, targetdb, catalogdb

* target: to run targeting

* design: to create designs

* scheduler: software to run scheduler

* kaiju: software to assign and test fiber configurations

* observesim: software to run simulations of above

Databases
---------

* catalogdb: input catalog database (heavy-weight)

* targetdb: target catalog database (medium-weight)

* fpsdb: operational database (light-weight)

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
 
