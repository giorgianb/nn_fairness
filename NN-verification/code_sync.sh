#!/bin/bash

USER="xingzguo"

LOCAL_DIR="/Users/xingzhiguo/Documents/git_project/NN-verification"
SERVER="pangu"
REMOTE_DIR="/home/xingzguo/hdd-space/git_project"  # pangu 
# SERVER="ai"
# REMOTE_DIR="/home/xingzguo/projects/"   # ai

# Open VSCode with your files.
code $LOCAL_DIR

echo "starting sync $LOCAL_DIR ---> $USER@$SERVER:$REMOTE_DIR"
# Start fswatch, wait for changes in your local dir and then execute rsync.
# NOTE: Syncs remotely as apache user.
#fswatch -o $LOCAL_DIR | xargs -n1 -I{} rsync -avzh -e "ssh" --rsync-path="rsync" $LOCAL_DIR $USER@$SERVER:$REMOTE_DIR"
fswatch -o $LOCAL_DIR | xargs -n1 -I{} sh -c "rsync -avzh -e 'ssh' --rsync-path='rsync' $LOCAL_DIR $USER@$SERVER:$REMOTE_DIR"

#rsync -r -v --progress -e ssh xingzguo@ai:/home/xingzguo/projects_data/news_headline_change/results ./