#!/usr/bin/bash

time_buckets=("all" "0" "1-12" "13-24" "25-36" "37-48" "49-60" "61-72" "73-84" "85-96" "97-108" "109-120" "121-132" "133-144" "145-156" "157-168" "169-180" "181-192" "193-204" "205-216" "217-228" "229-242")
declare -A dict
dict=(["t1"]=1 ["t2"]=1 ["ct1"]=1 ["flair"]=1 ["t1_t2"]=2 ["ct1_flair"]=2)

read -p "Modality set: " modality
read -p "Resolution: " resolution


for bucket in "${time_buckets[@]}"
do
    echo python test.py --modality_set "$modality" --input_C "${dict[$modality]}" --resolution "$resolution" --time_bucket "$bucket"
done