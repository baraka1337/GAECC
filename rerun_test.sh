if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <container_image>"
    exit 1
fi
while ! srun -p low --pty -N 1 --cpus-per-gpu=124 --mem=128G -G 1 --container-mounts $HOME:$HOME --container-image $1 --container-workdir=$HOME/GAECC_test python3.8 test_template.py; do
    echo "Command failed. Retrying..."
    sleep 1  # Optional: wait for a second before retrying
done

echo "Command succeeded."
exit 0
