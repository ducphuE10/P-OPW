import sys
import os

list_dir = os.listdir('./UCR_data/UCRArchive_2018')
print(list_dir)

for dataset in list_dir:
    print(dataset)
    os.system(f'python main.py --dataset {dataset}  \
        -tl1 0.3 \
        -tl2 0.3 \
        -m topw1 \
        -l1 50 \
        -l2 0.1 \
        --delta 1 \
        -m topw1 \
        --trend_method l1')