#!/usr/bin/bash

time_buckets=("all" "0" "1-10" "11-20" "21-30" "31-40" "41-50" "51-60" "61-70" "71-80" "81-90" "91-110" "111-130" "131-150" "151-170" "171-200")
declare -A dict
dict=(["t1"]=1 ["t2"]=1 ["ct1"]=1 ["flair"]=1 ["t1_t2"]=2 ["ct1_flair"]=2)

read -p "Modality set: " modality
read -p "Resolution: " resolution


for bucket in "${time_buckets[@]}"
do
    echo python test.py --modality_set "$modality" --input_C "${dict[$modality]}" --resolution "$resolution" --time_bucket "$bucket"
done