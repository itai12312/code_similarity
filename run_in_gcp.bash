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
# ssh -i ~/.ssh/id_rsa.pub zeitak_itai_gmail_com@34.76.125.245 (add ssh-keys via metadata)
# ls -la c_sharp_code | wc -l
# /usr/lib/p7zip/7z X c_sharp_code.7z -oc_sharp_code/ -aos
# --gcp_bucket jupter-data-for-sf --stages_to_run vectors tfidf --cores_to_use -1 --security_keywords sql xss xsrf xxe dos overflow injection cryptographically --matrix_form tfidf --vectorizer count --metric euclidean --input_folder ../codes/ --files_limit_end 100 --output_folder results1 --files_limit_step 30
# nohup main.py &
python3.6 main.py --shutdown --gcp_bucket jupter-data-for-sf --stages_to_run vectors tfidf --cores_to_use -1 --security_keywords sql xss xsrf xxe dos overflow injection cryptographically --matrix_form tfidf --vectorizer count --metric euclidean --input_folder ../codes/ --files_limit_end 130000 --output_folder results --files_limit_step 600 > logfile.txt 2>&1
sudo shutdown
# --cores_to_use 1 --security_keywords sql xss injection --matrix_form tfidf --vectorizer count --metric euclidean --input_folder ../codes_short/ --files_limit_end 10000 --output_folder result6 --files_limit_step 30