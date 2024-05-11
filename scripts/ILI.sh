#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
root_path=./Datasets
model_name=TFDNet
seq_len=96
mode=IK
individual_factor=7
for pred_len in 24 36 48 60
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path national_illness.csv \
	--model_id ILI_$seq_len'_'pred_len \
	--model $model_name \
	--data custom \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--mode $mode \
	--des 'Exp' \
	--enc_in 7 \
	--batch_size 32 \
	--lradj constant \
	--individual_factor $individual_factor \
	--train_epochs 50 \
	--dropout 0.01 \
	--patience 10 \
	--learning_rate 0.005 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ILI'_'$mode'_'$seq_len'_'$pred_len.log
done

seq_len=96
mode=MK
kernel_num=4
for pred_len in 24 36 48 60
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path national_illness.csv \
	--model_id ILI_$seq_len'_'pred_len \
	--model $model_name \
	--data custom \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--mode $mode \
	--des 'Exp' \
	--enc_in 7 \
	--batch_size 32 \
	--lradj constant \
	--kernel_num $kernel_num \
	--train_epochs 50 \
	--dropout 0.01 \
	--patience 10 \
	--learning_rate 0.005 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ILI'_'$mode'_'$seq_len'_'$pred_len.log
done