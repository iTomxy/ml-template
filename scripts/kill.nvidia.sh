nvidia-smi | awk '$3 ~/[0-9]+/ {if((NR>15)) {print $3}}' | xargs sudo kill -9
# nvidia-smi | awk '$5 ~/[0-9]+/ {if((NR>18)) {print $5}}' | xargs kill -9
nvidia-smi
