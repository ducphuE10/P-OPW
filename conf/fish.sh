#! /bin/sh
python main.py --dataset Fish  \
        -tl1 0.3 \
        -tl2 0.3 \
        -m topw1 \
        -l1 50 \
        -l2 0.1 \
        --delta 1 \
        -m topw1 \
        --trend_method l1

