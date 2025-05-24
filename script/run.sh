clear
# CUDA_VISIBLE_DEVICES=5 nohup python main.py > nk_unidepth.txt &
CUDA_VISIBLE_DEVICES=0 python main.py
#  fuser -v /dev/nvidia1