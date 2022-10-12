model=sbm
nc=128 # number of clusters
st=BMS+SB
sw=1e-4 # sparsity regularizer

for task in listops text retrieval image pathfinder32-curv_contour_length_14
do
    python3 run_tasks.py --task $task --model $model --num_clusters $nc --sbm_type $st --sparsity_weight $sw
done