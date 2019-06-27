# Kaggle-Jigsaw 

Our solution is described in [here](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/97425#latest-562308)

# Preprocessing 
### Extract data for BERT/GPT2/XLNET
```bash
bash bin/extract_data.sh
```

### Extract features (11 features) 
```bash
bash bin/extract_features.sh
```

### Create targets 
```bash
bash bin/extract_target.sh
```

# Train models
```bash 
seed=17493
depth=12 #11, 12 for Bert base, 23, 24 for Bert large
maxlen=220
batch_size=32
accumulation_steps=4
model_name=bert #gpt2, xlnet

CUDA_VISIBLE_DEVICES=3 python main_catalyst.py train    --seed=$seed \
                                                        --depth=$depth \
                                                        --maxlen=$maxlen \
                                                        --batch_size=$batch_size \
                                                        --accumulation_steps=$accumulation_steps \
                                                        --model_name=$model_name
```

# Predictions 
Change the settings as same as training phase. 
Ex: 
```
seed=17493
depth=12
maxlen=220
batch_size=32
accumulation_steps=4
model_name=bert
```

Then

```bash 
python make_submission.py
```
