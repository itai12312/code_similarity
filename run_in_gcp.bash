#!/usr/bin/env bash
# sshfs -o IdentityFile=/home/itai/.ssh/google_compute_engine,allow_other zeitak_itai_gmail_com@35.189.248.82:/home/zeitak_itai_gmail_com/ /home/itai/projects/gcpmounted/
# sshfs -o IdentityFile=/home/itai/.ssh/google_compute_engine,allow_other,sshfs_debug,debug,LOGLEVEL=DEBUG zeitak_itai_gmail_com@104.154.162.236:/home/zeitak_itai_gmail_com/projects/python/sl_dl/ /home/itai/projects/gcpmounted/
# nohup python3.6 lab2.py --plan ex5 --uploadtogcp > logfile.txt 2>&1 &
# gsutil cp gs://bucketname/ locallocation
# gcloud compute --project "clean-feat-590" ssh --zone "us-central1-c" "gpu"
# tail -f logfile.txt
# htop
# ps aux | grep "python3.6"
# python3.6 -m pip install -r requirements.txt --user
# export PYTHONPATH=yourlocation