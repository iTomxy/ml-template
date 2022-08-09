# https://blog.csdn.net/HackerTom/article/details/126257508

nvidia-smi | \
	grep -E "[0-9]+MiB\s*/\s*[0-9]+MiB" | \
	awk '{print ($9" "$11)}' | \
	sed "s/\([0-9]\{1,\}\)MiB \([0-9]\{1,\}\)MiB/\1 \2/" | \
	awk '{print $2 - $1}'
