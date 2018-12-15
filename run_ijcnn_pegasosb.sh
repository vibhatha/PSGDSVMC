parallelism=$1
batch=$2
alpha=$3
error=$4

sh run.sh ${parallelism} ijcnn1 22 35000 7000 ${batch} ${alpha} ${error}
