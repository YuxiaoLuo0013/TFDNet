#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=TFDNet

root_path=./Datasets
seq_len=720
mode=IK
individual_factor=7
for pred_len in 96 192 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm2.csv \
	--model_id ETTm2_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTm2 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--mode $mode \
	--individual_factor $individual_factor \
	--des 'Exp' \
	--enc_in 7 \
	--batch_size 128 \
	--train_epochs 50 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTm2'_'$mode'_'$seq_len'_'$pred_len.log
done


seq_len=720
mode=MK
kernel_num=4
for pred_len in 96 192 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm2.csv \
	--model_id ETTm2_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTm2 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--mode $mode \
	--kernel_num $kernel_num \
	--des 'Exp' \
	--enc_in 7 \
	--batch_size 128 \
	--train_epochs 50 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTm2'_'$mode'_'$seq_len'_'$pred_len.log
done
