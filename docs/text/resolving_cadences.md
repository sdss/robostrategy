A basic task that needs to be performed is to decide what cadence to
take in each field. This task is closely related to deciding which
fibers to assign to which target for each design. 

Here we treat the cadences as defined by a set of epochs, each of
which has number of exposures, a "delta" number of days since the
previous epoch, and a "delta_min" and "delta_max" associated with
it. The first exposure on the list has delta=0.

Cadence Consistency
===================

For a given field, it will have some cadence. For each robot in the
field, it can reach certain targets, which have some cadence
requirements.

We need to know the answer to the question:

 "Is cadence i achievable in a field with cadence j?"

For example, a single exposure cadence is always achievable, but if we
need two exposures on a target in cadence i then its achievability
depends on the requirements on their cadence and what is on offer
under cadence j. That is, if cadence i needs a month separation, but
cadence j has yearly separations, then it won't work. 

If we can determine this, we can construct a "cadence consistency
matrix", C_{ij} which contains the answer to the above question
encoded as a 0 or 1 (i.e. False or True). Note that this matrix is
decidedly not symmetric. This matrix will be the basis on which we
make decisions about which targets the robots in a given field with a
chosen cadence should look for.

To determine the cadence consistency, we have to look at the
individual epochs. Imagine we want to get the cadence specified in
cadence i. Well, the first exposure in the list needs to correspond to
some exposure in cadence j. So we will check each possibility in turn.
For each possibility, if it doesn't fit the lunation requirement or
the requirement on the number of exposures in that epoch, we move
on. If it does, we proceed recursively; that is, we assume that epoch
has been satisfied, and we search the remaining epochs of j for
solutions of the remaining epochs of i.

As this recursion happens, we collect the full solutions that work and
return them along with whether the cadence i fits into j.

Packing Targets into a Cadence
==============================

For each robot in a field observed under a certain cadence, there are
then some set of targets available to it. We want then to pack those
targets into the field cadence in such a way as to maximize the
science (i.e. by getting the most targets, or something).

We define A_{ijk}, where i indexes the target, j indexes all the
solutions for the target i cadence under the field cadence, and k
indexes the epoch. Note this isn't quite a 3D matrix, because there
are different numbers J of patterns for each target cadence. A_{ijk}
is tells you for target i, under pattern j, at epoch k, how many
exposures will be taken.

We then define:

 nexposure_k = number of exposures in epoch k of field cadence

Then you need to find w_{ijk}, which is the number of exposures for
target i under solution j in epoch k, such that:

 w_{ijk} == 0 or A_{ijk}
 \sum_{ij} w_{ijk} <= nexposure_k
 \sum_{j} [(\sum_k w_{ijk}) > 0] <= 1 (only one cadence j chosen for a target)
 w_{ijk} = w_{ijk'} for any choices k, k' (all epochs in a cadence, or none)

If you assign a value to each target i, you can define a total value:

 V = \sum_i v_i [(\sum_jk w_{ijk}) > 0] 

Constraint programming techniques can then find solution w_{ijk} that
maximize the value, and you then know for each epoch k which target i
to observe.
