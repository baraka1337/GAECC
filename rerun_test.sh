while ! srun -p low --pty -N 1 --cpus-per-gpu=124 --mem=128G -G 1 --container-mounts $HOME:$HOME --container-image ~/gaecc-docker-1.0-py3.8-tf2.4.0.sqsh --container-workdir=$HOME/GA python3.8 test_template.py; do
    echo "Command failed. Retrying..."
    sleep 1  # Optional: wait for a second before retrying
done

echo "Command succeeded."