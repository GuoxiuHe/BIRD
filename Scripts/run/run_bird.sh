# train
# CUDA_VISIBLE_DEVICES=2 nohup python ./main.py --phase train --model_name bird --data_name ppdd --memory 0.45 --suffix .try0 &
# CUDA_VISIBLE_DEVICES=2 nohup python ./main.py --phase train --model_name bird --data_name ppdd --memory 0.45 --suffix .try1 &

# local evaluation
# CUDA_VISIBLE_DEVICES=1 nohup python ./main.py --phase evaluate_val --model_name bird --data_name ppdd --memory 0 &
# CUDA_VISIBLE_DEVICES=1 nohup python ./main.py --phase evaluate_test --model_name bird --data_name ppdd --memory 0 &

# online testing
CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase predict_items --model_name bird --data_name ppdd --memory 0 &
CUDA_VISIBLE_DEVICES=0 nohup python ./main.py --phase predict_items2 --model_name bird --data_name ppdd --memory 0 &
