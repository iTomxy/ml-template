#!/bin/bash
# Kill my processes sieved by the given filter string.
#   E.g.: [bash] fkill.sh main.py
# If the filter string is not passed, kill all.
#   E.g.: [bash] fkill.sh

# filter string
fs=$1

if [ -z "$fs" ]; then
    ps aux | grep `whoami` | awk '{print $2}' | xargs kill -9
else
    ps aux | grep `whoami` | grep $fs | awk '{print $2}' | xargs kill -9
fi
