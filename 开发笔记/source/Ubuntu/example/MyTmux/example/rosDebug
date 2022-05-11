#!/bin/bash
# create a new session called ros
tmux new -s rosDebug -d
tmux split -v
tmux split -h
tmux selectp -t 1
tmux split -h

tmux send -t 1 'watch -n 2 rosnode list' ENTER
tmux send -t 2 'watch -n 2 rostopic list' ENTER

tmux a