

DEMO=$1 # i.e. dadi_joint or dadi_joint_mig
FOLDER=$2 # code folder

for SEED in 0 1 2 3 4
do
    # summary stats
    echo "python3 $2/summary_stats_multi.py output${SEED}.txt output${SEED}.pdf ${DEMO}"
    python3 $2/summary_stats_multi.py output${SEED}.txt output${SEED}.pdf ${DEMO}

    # loss plot
    echo "python3 $2/plotting/plot_loss.py -i output${SEED}.txt -o output${SEED}_loss.pdf"
    python3 $2/plotting/plot_loss.py -i output${SEED}.txt -o output${SEED}_loss.pdf
done
