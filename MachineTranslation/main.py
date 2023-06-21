import torch
import spacy
import numpy as np
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from Encoder import  Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq
from torch.utils.tensorboard import SummaryWriter


spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)

TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True)


train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))

#Create vocabolary

SRC.build_vocab(train_data, max_size = 10000, min_freq=2)
TRG.build_vocab(train_data,max_size = 10000,min_freq=2)




#Training hyperparameters

num_epochs = 10
learning_rate = 0.01
batch_size = 64

#Model hyparparameters

load_model = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoders = len(SRC.vocab)
input_size_decoder = len(TRG.vocab)
output_size = len(TRG.vocab)
encoder_embedding_size = 100
decoder_embedding_size = 100
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

#Tensorboard

writer = SummaryWriter(f'runs/loss_plot')
step=0

train_iterator, valid_iterator,test_iterator =BucketIterator.splits(
    (train_data,valid_data,test_data),
    batch_size = batch_size,
    sort_within_batch = True,
    sort_key = lambda x: len(x.src),device=device)

encoder_net = Encoder(input_size_encoders,encoder_embedding_size,hidden_size,num_layers,enc_dropout).to(device)

decoder_net = Decoder(input_size_decoder,decoder_embedding_size,hidden_size,output_size,num_layers,dec_dropout).to(device)

model = Seq2Seq(encoder_net,decoder_net,output_size,device).to(device)


pad_idx = TRG.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters())


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint")
    torch.save(state, filename)


def translate_sentence(model, sentence, src_field, trg_field, device, max_len=10):
    model.eval()
    if isinstance(sentence, str):
        tokens = tokenize_de(sentence)
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)

        with torch.no_grad():
            output, hidden, _ = model.decoder(trg_tensor, hidden, encoder_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]
    return trg_tokens[1:]  # Remove the initial <sos> token
def load_checkpoint(checkpoint):
    print("loading checkpoint:")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"))

examp_sentence = "Ich liebe dich so sehr"
for epoch in range(num_epochs):
    losses = []
    print(f"Epoch [{epoch} / {num_epochs}]")
    checkpoint = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    save_checkpoint(checkpoint)
    model.eval()
    translation = translate_sentence(model, examp_sentence, SRC, TRG, device)
    translated_sentence = " ".join(translation)

    print(f"Translation: {translated_sentence}")

    model.train()
    for batch_indx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)
        losses.append(loss.item())
        loss.backward()
        writer.add_scalar("Training Loss", loss, global_step=step)
        step += 1

    print(f"Mean loss: {sum(losses) / len(losses)}")

