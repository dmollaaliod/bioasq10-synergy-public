"NN models for classification"
import numpy as np
from multiprocessing import Pool
from functools import partial

from transformers import BertModel, BertTokenizer
from transformers import DistilBertModel, DistilBertTokenizer
from transformers import AlbertModel, AlbertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import progressbar

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using", device, "device")

def sentence_to_qbert_id(X_Q, tokenizer, max_length):
    result = tokenizer(X_Q[0], X_Q[1], add_special_tokens=True, 
                                           max_length=max_length,
                                           padding='max_length',
                                           truncation=True)

    if 'token_type_ids' not in result:
        # This part is needed because DistilBERT doesn't use token_type_ids
        segment_start = result['input_ids'].index(102) + 1
        if 0 in result['attention_mask']:
            segment_end = result['attention_mask'].index(0) - 1
        else:
            segment_end = len(result['attention_mask']) - 1
        #print(result)    
        #print(result['input_ids'][segment_start:segment_end])

        result['token_type_ids'] = [0] * segment_start + [1] * (segment_end - segment_start) + [0] * (len(result['input_ids']) - segment_end)
        assert len(result['token_type_ids']) == len(result['input_ids'])


    return result
    

def parallel_sentences_to_qbert_ids(batch_X, batch_Q, 
                                   sentence_length,
                                   tokenizer):
    "Convert the text to indices and pad-truncate to the maximum number of words"
    assert len(batch_X) == len(batch_Q)
    with Pool() as pool:
        result = pool.map(partial(sentence_to_qbert_id,
                                tokenizer=tokenizer,
                                max_length=sentence_length),
                        zip(batch_Q, batch_X))


    return {'input_ids': [x['input_ids'] for x in result],
            'attention_mask': [x['attention_mask'] for x in result],
            'token_type_ids': [x['token_type_ids'] for x in result]}

class BERT(nn.Module):
    """Simplest possible BERT classifier"""
    def __init__(self, batch_size=32, hidden_layer_size=0, dropout_rate=0.5, 
                 positions=False, finetune_model=False):
        super(BERT, self).__init__()
        self.batch_size = batch_size
        self.sentence_length = 250 # as observed in exploratory file bert-exploration.ipynb
        self.hidden_layer_size = hidden_layer_size
        self.tokenizer = self.load_tokenizer()
        self.cleantext = lambda t: t
        self.positions = positions
        self.finetune = finetune_model

        # Model layers   
        bert_config, self.bert_layer = self.load_bert_model()
        for param in self.bert_layer.parameters():
            param.requires_grad = False

        self.bert_hidden_size = bert_config['hidden_size']

        if self.positions:
            bert_size = self.bert_hidden_size + 1

        else:
            bert_size = self.bert_hidden_size

        if hidden_layer_size > 0:
            self.hidden_layer = nn.Linear(bert_size, 
                                          hidden_layer_size)
            output_size = hidden_layer_size
        else:
            output_size = bert_size
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(output_size, 1)

        self.to(device)

        print("Number of parameters:",
              sum(p.numel() for p in self.parameters()))
        print("Number of trainable parameters:",
              sum(p.numel() for p in self.parameters() if p.requires_grad))

    def save(self, savepath):
        torch.save(self.state_dict(), savepath)

    def load(self, loadpath):
        self.load_state_dict(torch.load(loadpath))

    def forward(self, input_word_ids, input_masks, input_segments, avg_mask, positions):

        # BERT
        if self.name().startswith('DistilBERT'):
            #print("We are in DistilBERT")
            bert = self.bert_layer(input_ids=input_word_ids, 
                                   attention_mask=input_masks)
        else:
            #print("We are NOT in DistilBERT")
            bert = self.bert_layer(input_ids=input_word_ids, 
                                   attention_mask=input_masks, 
                                   token_type_ids=input_segments)

        # Average using avg_mask
        mask = avg_mask.unsqueeze(2).repeat(1, 1, self.bert_hidden_size)
        inputs_sum = torch.sum(bert.last_hidden_state * mask, 1)
        pooling = inputs_sum / torch.sum(mask, 1)

        # Sentence positions
        if self.positions:
            #print(pooling.shape)
            #print(positions.shape)
            all_inputs = torch.cat([pooling, positions], 1)

        else:
            all_inputs = pooling

        # Hidden layer
        if self.hidden_layer_size > 0:
            hidden = F.relu(self.hidden_layer(all_inputs))
        else:
            hidden = all_inputs
        dropout = self.dropout_layer(hidden)

        # Final outputs
        outputs = torch.sigmoid(self.output_layer(dropout))

        return(outputs)

    def load_tokenizer(self):
        return BertTokenizer.from_pretrained('bert-base-uncased')

    def load_bert_model(self):
        return {'hidden_size': 768}, BertModel.from_pretrained('bert-base-uncased')
 
    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "BERT%s%s%s" % (str_finetune, str_positions, str_hidden)

    def fit(self, X_train, Q_train, Y_train,
            X_positions = [],
            validation_data=None,
            verbose=2, nb_epoch=3,
            savepath=None, restore_model=False):

        # Training loop
        X = parallel_sentences_to_qbert_ids(X_train, Q_train, self.sentence_length, self.tokenizer)

        return self.__fit__(X, X_positions, Y_train,
                            validation_data,
                            verbose, nb_epoch, None, savepath)

    def __fit__(self, X, X_positions, Y,
                validation_data,
                verbose, nb_epoch, learning_rate,
                savepath=None):

        criterion = nn.BCELoss()
        if learning_rate:
            optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        else:
            optimizer = optim.Adam(self.parameters())

        history = {'loss':[], 'val_loss':[]}

        print("Batch size:", self.batch_size)
        print("Input data size:", len(X['input_ids']))

        self.train()
        for epoch in range(nb_epoch):
            if self.finetune and epoch == 1:
                print("Unfreezing the model")
                for param in self.bert_layer.parameters():
                    param.requires_grad = True
                for param in list(self.bert_layer.embeddings.parameters()):
                    param.requires_grad = False

                print("Number of parameters:",
                sum(p.numel() for p in self.parameters()))
                print("Number of trainable parameters:",
                sum(p.numel() for p in self.parameters() if p.requires_grad))


            #print("Entering epoch", epoch)
            # running_loss = 0.0
            epoch_loss = 0.0
            batch_count = 0
            progress = progressbar.ProgressBar(max_value=len(X['input_ids']) // self.batch_size,
                                               variables={'loss': '--'},
                                               suffix=' Loss: {variables.loss}',
                                               redirect_stdout=True).start()
            for batch in range(len(X['input_ids']) // self.batch_size):
                f = self.batch_size*batch
                t = self.batch_size*(batch+1)
                input_word_ids = torch.tensor(X['input_ids'][f:t], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:t], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:t], device=device)
                input_positions = torch.tensor(X_positions[f:t], device=device).float()
                avg_mask = torch.tensor(X['token_type_ids'][f:t], device=device)
                targets = torch.tensor(Y[f:t], device=device)#.unsqueeze(1)

                #print("Size of inputs:", input_word_ids.size())

                optimizer.zero_grad()
                outputs = self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)
                #print("Size of outputs:", outputs.size())
                #print("Size of targets:", targets.size())
                loss = criterion(outputs.float(), targets.float())
                loss.backward()
                optimizer.step()

                # running_loss += loss.item()
                epoch_loss += loss.item()
                batch_count += 1

                # modulo = 1 # Report after this number of batches
                # if batch % modulo == modulo - 1:
                #     print('[%d, %5d] loss: %.3f' %
                #           (epoch + 1, batch + 1, running_loss / modulo))
                #     running_loss = 0.0
                progress.update(batch+1, loss="%.4f" % loss.item())

            history['loss'].append(epoch_loss / batch_count)

            if validation_data:
                history['val_loss'].append(self.test(*validation_data)['loss'])
                print("Epoch", epoch+1, "loss:", history['loss'][-1], 
                      "validation loss:", history['val_loss'][-1])
            else:
                print("Epoch", epoch+1, "loss:", history['loss'][-1])


        return history


    def predict(self, X_topredict, Q_topredict, X_positions=[]):
        X = parallel_sentences_to_qbert_ids(X_topredict, Q_topredict, self.sentence_length, self.tokenizer)
        # input_word_ids = torch.tensor(X['input_ids'], device=device)
        # input_masks = torch.tensor(X['attention_mask'], device=device)
        # input_segments = torch.tensor(X['token_type_ids'], device=device)
        # input_positions = torch.tensor(X_positions, device=device).float()
        # avg_mask = torch.tensor(X['token_type_ids'], device=device)

        progress = progressbar.ProgressBar(max_value=len(X['input_ids']) // self.batch_size,
                                               redirect_stdout=True).start()

        result = []

        self.eval()
        with torch.no_grad():
            for batch in range(len(X['input_ids']) // self.batch_size):
                f = self.batch_size*batch
                t = self.batch_size*(batch+1)
                input_word_ids = torch.tensor(X['input_ids'][f:t], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:t], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:t], device=device)
                input_positions = torch.tensor(X_positions[f:t], device=device).float()

                avg_mask = torch.tensor(X['token_type_ids'][f:t], device=device)

                result += self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)

                progress.update(batch+1)

            if len(X['input_ids']) % self.batch_size != 0:
                # Remaining data 
                f =  len(X['input_ids']) // self.batch_size * self.batch_size
                input_word_ids = torch.tensor(X['input_ids'][f:], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:], device=device)
                input_positions = torch.tensor(X_positions[f:], device=device).float()

                avg_mask = torch.tensor(X['token_type_ids'][f:], device=device)

                result += self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)

        return result

    def test(self, X_test, Q_test, Y_test, X_positions=[]):
        X = parallel_sentences_to_qbert_ids(X_test, Q_test, self.sentence_length, self.tokenizer)
        return self.__test__(X, Y_test, X_positions)

    def __test__(self, X,
                       Y,
                       X_positions=[]):

        criterion = nn.BCELoss()
        test_loss = 0.0
        loss_count = 0
        progress = progressbar.ProgressBar(max_value=len(X['input_ids']) // self.batch_size,
                                               variables={'loss': '--'},
                                               suffix=' Loss: {variables.loss}',
                                               redirect_stdout=True).start()

        self.eval()
        with torch.no_grad():
            for batch in range(len(X['input_ids']) // self.batch_size):
                f = self.batch_size*batch
                t = self.batch_size*(batch+1)
                input_word_ids = torch.tensor(X['input_ids'][f:t], device=device)
                input_masks = torch.tensor(X['attention_mask'][f:t], device=device)
                input_segments = torch.tensor(X['token_type_ids'][f:t], device=device)
                input_positions = torch.tensor(X_positions[f:t], device=device).float()

                avg_mask = torch.tensor(X['token_type_ids'][f:t], device=device)
                targets = torch.tensor(Y[f:t], device=device)#.unsqueeze(1)

                outputs = self(input_word_ids, input_masks, input_segments, avg_mask, input_positions)

                loss = criterion(outputs.float(), targets.float())

                test_loss += loss.item()
                loss_count += 1

                progress.update(batch+1, loss="%.4f" % loss.item())

        return {'loss': test_loss/loss_count}


class BioBERT(BERT):
    """A simple BERT system that uses the BioBERT weights"""
    def load_tokenizer(self):
        return BertTokenizer.from_pretrained("monologg/biobert_v1.1_pubmed") # , cache_dir="huggingface_models")
#        return BertTokenizer.from_pretrained('biobert_v1.1_pubmed')

    def load_bert_model(self):
        return {'hidden_size': 768}, BertModel.from_pretrained("monologg/biobert_v1.1_pubmed") #, cache_dir="huggingface_models")
#        return {'hidden_size': 768}, BertModel.from_pretrained('biobert_v1.1_pubmed')

    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "BioBERT%s%s%s" % (str_finetune, str_positions, str_hidden)


class DistilBERT(BERT):
    """A simple DistilBERT system"""
    def load_tokenizer(self):
        return DistilBertTokenizer.from_pretrained("distilbert-base-uncased") # , cache_dir="huggingface_models")

    def load_bert_model(self):
        return {'hidden_size': 768}, DistilBertModel.from_pretrained("distilbert-base-uncased") #, cache_dir="huggingface_models")

    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "DistilBERT%s%s%s" % (str_finetune, str_positions, str_hidden)

class ALBERT(BERT):
    """Using ALBERT"""
    def load_tokenizer(self):
        return AlbertTokenizer.from_pretrained("albert-xxlarge-v2" , cache_dir="huggingface_models")

    def load_bert_model(self):
        return {'hidden_size': 4096}, AlbertModel.from_pretrained("albert-xxlarge-v2", cache_dir="huggingface_models")

    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "ALBERT%s%s%s" % (str_finetune, str_positions, str_hidden)

class ALBERT_squad2(BERT):
    """Using ALBERT"""
    def load_tokenizer(self):
        return AlbertTokenizer.from_pretrained("mfeb/albert-xxlarge-v2-squad2" , cache_dir="huggingface_models")
#        return AlbertTokenizer.from_pretrained("mfeb_albert-xxlarge-v2-squad2")

    def load_bert_model(self):
        return {'hidden_size': 4096}, AlbertModel.from_pretrained("mfeb/albert-xxlarge-v2-squad2", cache_dir="huggingface_models")
#        return {'hidden_size': 4096}, AlbertModel.from_pretrained("mfeb_albert-xxlarge-v2-squad2")


    def name(self):
        if self.positions:
            str_positions = "+pos"
        else:
            str_positions = ""
        if self.hidden_layer_size == 0:
            str_hidden = ""
        else:
            str_hidden = "-relu(%i)" % self.hidden_layer_size
        if self.finetune:
            str_finetune = "(finetuned)"
        else:
            str_finetune = "(frozen)"
        return "ALBERT_squad2%s%s%s" % (str_finetune, str_positions, str_hidden)
