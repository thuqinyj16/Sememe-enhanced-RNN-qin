# Sememe-enhanced Recurrent Neural Networks
This is the implementation of "sememe-enhanced neural networks", if you have any questions, feel free to contact me: qinyj16@mails.tsinghua.edu.cn :)


## Language model

## Sentence encoders (test on SNLI)

First please download the pretrained glove embeddings, which can be achieved through: https://nlp.stanford.edu/projects/glove/ 

Then you could easily run the code by:

```
python3 train_nli.py --word_emb_path ../glove/glove.840B.300d.txt --encoder_type LSTM_sememe --gpu_id 2
```

All the models are in the FILE:models, you can replace data.py & models.py to experiment on another model, notice that you may have to change the --encoder_type & --word_emb_path hyper-parameters.
