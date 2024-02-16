# https://blog.csdn.net/HackerTom/article/details/136127369
param ($b = 50)
# echo $b
(Get-WmiObject -Namespace root/WMI -Class WmiMonitorBrightnessMethods).WmiSetBrightness(1,$b)
