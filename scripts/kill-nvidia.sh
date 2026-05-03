# kill all processes launched by me which are using GPU
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill -9
