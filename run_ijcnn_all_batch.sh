declare -a nodes=("32" "16" "8" "4" "2")
declare -a batches=("1" "2" "4" "8" "16")
declare -a errors=("0.20")

## now loop through the above array
for i in "${nodes[@]}"
do
   for j in "${batches[@]}"
   do
     for k in "${errors[@]}"
     do 
          sh run_ijcnn_pegasosb.sh $i $j 0.001 $k 
     done   
  done   
done
#sh run_ijcnn1_pegasosb.sh 32 1 0.001 0.20
