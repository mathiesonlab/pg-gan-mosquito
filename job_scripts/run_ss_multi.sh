

DEMO=$1 # i.e. dadi_joint or dadi_joint_mig
FOLDER=$2 # code folder
PREFIX=$3 # i.e. CI_MZ_dadi_joint

for SEED in 0 1 2 3 4 5 6 7 8
do
    # summary stats
    echo "python3 $2/summary_stats_multi.py ${PREFIX}${SEED}.txt ${PREFIX}${SEED}.pdf ${DEMO}"
    python3 $2/summary_stats_multi.py ${PREFIX}${SEED}.txt ${PREFIX}${SEED}.pdf ${DEMO}

    # loss plot
    #echo "python3 $2/plotting/plot_loss.py -i ${PREFIX}${SEED}.txt -o ${PREFIX}${SEED}_loss.pdf"
    #python3 $2/plotting/plot_loss.py -i ${PREFIX}${SEED}.txt -o ${PREFIX}${SEED}_loss.pdf
done
