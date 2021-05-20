#! /bin/bash

epochs=30001
batch_size=32
ch_scale=1.5

#models_10_30
#models_80_100
#models_150_170
n_masks=1
sinterval=200

part=dpt-gpu-EL7
part=dpt-EL7

##--analyze --restart
moreflags="--analyze"
#moreflags=""
#10 80 150 220 290 


ll=64
kernel=3 

for alpha in 0.01 0.05 0.1 0.2 ; do

for amin in 290 ; do
for nlg in 3 ; do 

#vsub -c "python ganQ.py --nlg $nlg --epochs $epochs --amin $amin --alpha $alpha $moreflags" --time 12:00:00 --part $part --mem 25000 -N 5 --name train-$alpha-$amin-$nlg

for fnl in 0 5 10 100 1000; do
#vsub -c "python gan2.py --nlg $nlg --epochs $epochs --amin $amin --alpha $alpha $moreflags --nfl $fnl" --time 12:00:00 --part $part --mem 25000 --name train-$alpha-$amin-$nlg
vsub -c "python test3.py --nlg $nlg --alpha $alpha --nfl $fnl --mamin $amin --tamin 1 --tamax 1000" --time 12:00:00 --part $part --mem 25000 -N 4 --name test-$alpha-$amin-$nlg
done

#for tamax in 100 200 500 1000 1500 2000; do 
#vsub -c "python test2.py --alpha $alpha --nlg $nlg --mamin $amin --tamin 1 --tamax $tamax" \
#--time 12:00:00 --part $part --mem 30000 -N 8 --name test-$alpha-$amin-$nlg-$tamax
#sleep 5
#done


done
done
done












#models_80_100/l64_g5_d4_ch1.5_k3

#declare -a arr=("models_150_170/l64_g3_d2_ch1.5_k3"
#                "models_220_240/l64_g5_d4_ch1.5_k3"
#                "models_290_310/l64_g3_d2_ch1.5_k3"
#                "models_290_310/l64_g4_d3_ch1.5_k3"
#                "models_290_310/l64_g5_d4_ch1.5_k3"
#                )

#for i in "${arr[@]}"; do

#rm -r $i/model
#rm $i/images/3????.png
#rm $i/images/4????.png
#rm $i/images/5????.png

#cp -r backup/$i/model $i/model

#done





#vsub -c "python gan1.py --nlg 5 --epochs $epochs --amin 80 $moreflags" --time 12:00:00 --part $part --mem 15000 --name m0

#vsub -c "python gan1.py --nlg 3 --epochs $epochs --amin 150 $moreflags" --time 12:00:00 --part $part --mem 15000 --name m1

#vsub -c "python gan1.py --nlg 5 --epochs $epochs --amin 220 $moreflags" --time 12:00:00 --part $part --mem 15000 --name m2

#vsub -c "python gan1.py --nlg 3 --epochs $epochs --amin 290 $moreflags" --time 12:00:00 --part $part --mem 15000 --name m3

#vsub -c "python gan1.py --nlg 4 --epochs $epochs --amin 290 $moreflags" --time 12:00:00 --part $part --mem 15000 --name m4

#vsub -c "python gan1.py --nlg 5 --epochs $epochs --amin 290 $moreflags" --time 12:00:00 --part $part --mem 15000 --name m5



#for nld in 2 3 4; do 
#nlg=$((nld+1))
#vsub -c "python gan1.py --lside 64 \
#--nld $nld --nlg $nlg --kernel 3 --epochs 30001 --batch_size 32 \
#--ch_scale $ch_scale --amin 100 --amax 1000 --n_masks 1 --sinterval 200 $moreflags" \
#--time 12:00:00 --part $part --mem 15000 \
#--name large-$nlg
#done





#for ii in {0..4}; do
#for amin in 80 150 220 290 ; do
#for ll in 64 ; do
#for nld in 2 3 4 ; do 
#for kernel in 3 ; do

#nlg=$((nld+1))
#amax=$((amin+20))

##1 5 10
##10 20 100

##for tamin in 1 5 10; do
##for tamax in 15 100 1000; do 
##vsub -c "python test.py --lside $ll \
##--nld $nld --nlg $nlg --kernel $kernel --ch_scale $ch_scale --mamin $amin \
##--tamin $tamin --tamax $tamax" \
##--time 12:00:00 --part $part --mem 15000 \
##--name $ll-$nld-$nlg-$kernel-$ch_scale
##done
##done

#for tamax in 100 200 300 500 1000 1500 2000; do 
#vsub -c "python test.py --num $ii --nlg $nlg --mamin $amin \
#--tamin 1 --tamax $tamax" \
#--time 12:00:00 --part $part --mem 15000 \
#--name $ll-$nld-$nlg-$kernel-$ch_scale
#done

#done
#done
#done
#done

#done






#zip res.zip {models_*/*/cl*,models_*/*.png,models_*/*.npy,models_*/*/power.jpg,,models_*/*/model/nres.npy,res/*}

#zip backup.zip {models_*/*/cl*,models_*/*.png,models_*/*.npy,models_*/*/power.jpg,,models_*/*/model/*,res/*}

#vsub -c "python null.py --lside 64 --amin 10 --n_masks 1" --mem 12000 --time 4:00:00










