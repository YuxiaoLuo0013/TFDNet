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
individual_factor=64
pred_len=96
python -u run_longExp.py \
--is_training 1 \
--data_path electricity.csv \
--model_id electricity_$seq_len'_'$pred_len'_'$n_channel'_'$mode \
--model $model_name \
--data custom \
--root_path $root_path \
--seq_len $seq_len \
--pred_len $pred_len \
--des 'Exp' \
--enc_in 321 \
--batch_size 32 \
--dropout 0.01 \
--kernel_size 25 \
--individual_factor $individual_factor \
--mode $mode \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.0001 \
--itr 3 >logs/LongForecasting/TDFNet'_'ECL'_'$mode'_'$seq_len'_'$pred_len.log

seq_len=720
for pred_len in 192 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path electricity.csv \
	--model_id electricity_$seq_len'_'$pred_len'_'$n_channel'_'$mode \
	--model $model_name \
	--data custom \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 321 \
	--batch_size 128 \
	--dropout 0.01 \
	--kernel_size 25 \
	--individual_factor $individual_factor \
	--mode $mode \
	--train_epochs 100 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ECL'_'$mode'_'$seq_len'_'$pred_len.log
done

seq_len=512
mode=MK
kernel_num=16
python -u run_longExp.py \
--is_training 1 \
--data_path electricity.csv \
--model_id electricity_$seq_len'_'$pred_len'_'$n_channel'_'$mode \
--model $model_name \
--data custom \
--root_path $root_path \
--seq_len $seq_len \
--pred_len $pred_len \
--des 'Exp' \
--enc_in 321 \
--batch_size 32 \
--dropout 0.01 \
--kernel_size 25 \
--kernel_num $kernel_num \
--mode $mode \
--train_epochs 100 \
--patience 10 \
--learning_rate 0.0001 \
--itr 3 >logs/LongForecasting/TDFNet'_'ECL'_'$mode'_'$seq_len'_'$pred_len.log

seq_len=720
for pred_len in 192 336 720
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path electricity.csv \
	--model_id electricity_$seq_len'_'$pred_len'_'$n_channel'_'$mode \
	--model $model_name \
	--data custom \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 321 \
	--batch_size 128 \
	--dropout 0.01 \
	--kernel_size 25 \
	--kernel_num $kernel_num \
	--mode $mode \
	--train_epochs 100 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 >logs/LongForecasting/TDFNet'_'ECL'_'$mode'_'$seq_len'_'$pred_len.log
done