for model in softmax nystrom-256 linformer-256 reformer-2 performer-256 linear
do
    for task in listops text retrieval image pathfinder32-curv_contour_length_14
    do
        python3 run_tasks.py --task $task --model $model --suffix run_baselines_$model
    done
done
