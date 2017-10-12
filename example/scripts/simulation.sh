SAVEPATH='../results'

for epoch in 5 # 1 3 5
    do
    for alpha in 1 # 1 5 10 15 20
        do
        python script.py --init_exploration 30 \
            --lookback 5\
            --iterations 300\
            --epochs ${epoch}\
            --alpha ${alpha}\
            --save_path ${SAVEPATH}\
            --save \
            --no-reshape \
            --filename alpha${alpha}_epoch${epoch}_ \
            # --interactive
        done
    done
exit 0

