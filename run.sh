
for dim in 1000
do
    for levels in 10
    do
        for flipping in 0.005 0.01
        do
            for win_secs in 1.0 2.0 3.0 5.0
            do
                for overlap in 0.0
                do
                    for n_mels in 128
                    do
                        time python3 main.py --dataset $1 --dim $dim --levels $levels --flipping $flipping --win_secs $win_secs --overlap $overlap --n_mels $n_mels
                    done
                done
            done
        done
    done
done