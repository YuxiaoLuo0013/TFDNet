#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi

root_path=./Datasets
model_name=TFDNet
mode=IK
individual_factor=7
seq_len=512
for pred_len in 96 192 
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTh1.csv \
	--model_id ETTh1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTh1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--individual_factor $individual_factor \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--dropout 0.05 \
	--patience 10 \
	--learning_rate 0.00005 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTh1'_'$mode'_'$seq_len'_'$pred_len.log
done
seq_len=720
for pred_len in 336 720 
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTh1.csv \
	--model_id ETTh1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTh1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--individual_factor $individual_factor \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--dropout 0.05 \
	--patience 10 \
	--learning_rate 0.00005 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTh1'_'$mode'_'$seq_len'_'$pred_len.log
done


seq_len=512
mode=MK
kernel_num=4
for pred_len in 96 192 
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTh1.csv \
	--model_id ETTh1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTh1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--kernel_num $kernel_num \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--dropout 0.05 \
	--patience 10 \
	--learning_rate 0.00005 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTh1'_'$mode'_'$seq_len'_'$pred_len.log
done
seq_len=720
for pred_len in 336 720 
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTh1.csv \
	--model_id ETTh1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTh1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--kernel_num $kernel_num \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--dropout 0.05 \
	--patience 10 \
	--learning_rate 0.00005 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTh1'_'$mode'_'$seq_len'_'$pred_len.log
done