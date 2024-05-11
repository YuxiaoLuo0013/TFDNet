if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=TFDNet
mode=MK
kernel_num=4
root_path=./Datasets
seq_len=336
for pred_len in 96 192 336
do
	python -u run_longExp.py \
	--is_training 1 \
	--data_path ETTm2.csv \
	--model_id ETTm2_$seq_len'_'$pre_len'_'$mode \
	--model $model_name \
	--data ETTm2 \
	--root_path $root_path \
	--seq_len $seq_len \
	--pred_len $pred_len \
	--des 'Exp' \
	--enc_in 1 \
	--kernel_num $kernel_num \
	--mode $mode \
	--batch_size 128 \
	--train_epochs 50 \
	--dropout 0.1 \
	--patience 10 \
	--learning_rate 0.0001 \
	--itr 3 --feature S >logs/LongForecasting/TDFNet'_'fS_ETTm2'_'$seq_len'_'$pred_len.log
done

seq_len=512
pred_len=720
python -u run_longExp.py \
--is_training 1 \
--data_path ETTm2.csv \
--model_id ETTm2_$seq_len'_'$pre_len'_'$mode \
--model $model_name \
--data ETTm2 \
--root_path $root_path \
--seq_len $seq_len \
--pred_len $pred_len \
--des 'Exp' \
--enc_in 1 \
--kernel_num $kernel_num \
--mode $mode \
--batch_size 128 \
--train_epochs 50 \
--dropout 0.1 \
--patience 10 \
--learning_rate 0.0001 \
--itr 3 --feature S >logs/LongForecasting/TDFNet'_'fS_ETTm2'_'$seq_len'_'$pred_len.log