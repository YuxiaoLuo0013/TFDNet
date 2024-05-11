#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=TFDNet
root_path=./Datasets
mode=IK
individual_factor=7
seq_len=336
for pred_len in 96 192
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm1.csv \
	--model_id L1ETTm1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTm1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--individual_factor $individual_factor \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTm1'_'$mode'_'$seq_len'_'$pred_len.log
done

seq_len=720
for pred_len in 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm1.csv \
	--model_id L1ETTm1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTm1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--individual_factor $individual_factor \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTm1'_'$mode'_'$seq_len'_'$pred_len.log
done

mode=MK
kernel_num=4
#
seq_len=336
for pred_len in 96 192
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm1.csv \
	--model_id L1ETTm1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTm1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--kernel_num $kernel_num \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTm1'_'$mode'_'$seq_len'_'$pred_len.log
done

seq_len=720
for pred_len in 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm1.csv \
	--model_id L1ETTm1_$seq_len'_'pre_len \
	--model $model_name \
	--data ETTm1 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 7 \
	--kernel_num $kernel_num \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ETTm1'_'$mode'_'$seq_len'_'$pred_len.log
done

