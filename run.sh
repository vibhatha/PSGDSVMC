parallelism=$1
dataset=$2
features=$3
trainingSamples=$4
testingSamples=$5
batch=$6
alpha=$7
error=$8

#echo "$parallelism $dataset $features $trainingSamples $testingSamples $batch $alpha $error"
echo "mpirun -n ${parallelism} ./openmP -dataset ${dataset} -features ${features} -trainingSamples ${trainingSamples} -testingSamples ${testingSamples} -itr 1000 -nt -pegasosb -batch_size ${batch} -split -ratio 0.80 -alpha ${alpha} -error ${error}"


mpirun -n ${parallelism} ./cmake-build-debug/PSGDC__ -dataset ${dataset} -features ${features} -trainingSamples ${trainingSamples} -testingSamples ${testingSamples} -itr 1000 -nt -pegasosb -batch_size ${batch} -split -ratio 0.80 -alpha ${alpha} -error ${error}
