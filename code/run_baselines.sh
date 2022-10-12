for model in softmax nystrom-256 linformer-256 reformer-2 performer-256 linear
do
    for task in listops text retrieval image pathfinder32-curv_contour_length_14
    do
        python3 run_tasks.py --task $task --model $model --dam_type none --dam_init none --dam_bp none --dam_dist1 none --dam_dist2 none --sparsity_weight 0.0 --suffix run_baselines_$model
    done
done