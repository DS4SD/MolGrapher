#!/usr/bin/bash

############################################

#TEMPLATE=dc-uspto-molgrapher.template
#TYPE='applications\/2022'
#YEAR=2022
#N_MODE=3

############################################

TEMPLATE=dc-uspto-molgrapher-cpu.template
TYPE='applications\/2022'
YEAR=2022
N_MODE=3

############################################

for job in `seq 0 $((N_MODE-1))`
do
    TYPED=$(dirname $TYPE)
    TYPED=${TYPED::-1}
    yaml=dc-$TYPED-$YEAR-$N_MODE-$job.yaml
    cp $TEMPLATE $yaml
    
    sed -i "s/__TYPED__/$TYPED/g"   $yaml
    sed -i "s/__TYPE__/$TYPE/g"   $yaml
    sed -i "s/__YEAR__/$YEAR/g"   $yaml
    sed -i "s/__N_MODE__/$N_MODE/g" $yaml
    sed -i "s/__I_MODE__/$job/g"    $yaml

done