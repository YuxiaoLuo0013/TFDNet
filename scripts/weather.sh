#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
root_path=./Datasets
model_name=TFDNet
seq_len=512
mode=IK
individual_factor=21
for pred_len in 96 192 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path weather.csv \
	--model_id weather_$seq_len'_'$pred_len'_'$mode \
	--model $model_name \
	--data custom \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--features M \
	--des 'Exp' \
	--enc_in 21 \
	--mode $mode \
	--individual_factor $individual_factor \
	--kernel_size 25 \
	--dropout 0.1 \
	--batch_size 128 \
	--train_epochs 100 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'weather'_'$mode'_'$seq_len'_'$pred_len.log
done 

mode=MK
kernel_num=4
for pred_len in 96 192 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path weather.csv \
	--model_id weather_$seq_len'_'$pred_len'_'$mode'_'abmoe \
	--model $model_name \
	--data custom \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--features M \
	--des 'Exp' \
	--enc_in 21 \
	--mode $mode \
	--kernel_num $kernel_num \
	--kernel_size 25 \
	--dropout 0.1 \
	--batch_size 128 \
	--train_epochs 100 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'weather'_'$mode'_'$seq_len'_'$pred_len.log
done 
