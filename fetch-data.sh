#! /bin/sh

cd data/external

wget https://physionet.org/pn6/chbmit/RECORDS-WITH-SEIZURES
for filename in $(cat RECORDS-WITH-SEIZURES)
do
  URLBASE="https://physionet.org/pn6/chbmit/"
  wget "$URLBASE$filename" -o recordings/$(basename $filename)
done
