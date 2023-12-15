echo "Conducting experiments on $1"
if [ "$1" = "PubMed" ]; then
    gam2=3 eps=0.05 alpha=0.6 beta=0.6 T=3
elif [ "$1" = "Cora" ]; then
    gam2=2 eps=0.05 alpha=0.6 beta=0.6 T=5
elif [ "$1" = "CiteSeer" ]; then
    gam2=2 eps=0.4 alpha=0.6 beta=0.7 T=4  
elif [ "$1" = "CoraFull" ]; then
    gam2=2 eps=0.01 alpha=0.7 beta=0.7 T=2 
elif [ "$1"="ogbn-arxiv" ]; then
    gam2=2 eps=.05 alpha=0.5 beta=0.6 T=50 
else 
    echo "Dataset not supported"
    exit 1
fi

for corrupt_ratio in 0.1 0.2 0.3 0.4 0.5; do
    if [ "$1" = "CoraFull" ]; then 
        python scripts/train_main_corafull.py --dataset $1 --corrupt_ratio $corrupt_ratio --gam2 $gam2 --eps $eps --alpha $alpha --beta $beta --T $T --corrupt_type asymm
        python scripts/train_main_corafull.py --dataset $1 --corrupt_ratio $corrupt_ratio --gam2 $gam2 --eps $eps --alpha $alpha --beta $beta --T $T --corrupt_type symm
    elif [ "$1" = "ogbn-arxiv" ]; then
        python scripts/train_main_arxiv.py --dataset $1 --corrupt_ratio $corrupt_ratio --gam2 $gam2 --eps $eps --alpha $alpha --beta $beta --T $T --corrupt_type asymm
        python scripts/train_main_arxiv.py --dataset $1 --corrupt_ratio $corrupt_ratio --gam2 $gam2 --eps $eps --alpha $alpha --beta $beta --T $T --corrupt_type symm
    else
        python scripts/train_main_planetoid.py --dataset $1 --corrupt_ratio $corrupt_ratio --gam2 $gam2 --eps $eps --alpha $alpha --beta $beta --T $T --corrupt_type asymm
        python scripts/train_main_planetoid.py --dataset $1 --corrupt_ratio $corrupt_ratio --gam2 $gam2 --eps $eps --alpha $alpha --beta $beta --T $T --corrupt_type symm
    fi
done
