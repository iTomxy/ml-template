import os

"""reference
- blog.csdn.net/m0_38007695/article/details/88954699
"""

pid = list(set(os.popen("fuser -v /dev/nvidia*").read().split()))
print("--- pid ---\n", pid)

kill_cmd = "kill -9 " + " ".join(pid)
print(kill_cmd)
os.popen(kill_cmd)

