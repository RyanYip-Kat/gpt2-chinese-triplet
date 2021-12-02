import io
import os
import torch
import argparse
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
#from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from tokenizations import tokenization_bert
from classify_dataset import CustomDataset,Gpt2ClassificationCollator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",type=str,default="train.txt",help="train text file:(text,label)")
    parser.add_argument("--valid",type=str,default="valid.txt",help="valid text file:(text,label)")
    parser.add_argument("--pretrained_model",type=str,default="gpt2-chinese-youlai/final_model/",help="gpt2 chinese pretrained model path")
    parser.add_argument("--vocab",type=str,default="gpt2-chinese-youlai/final_model/vocab.txt",help="vocab.txt file for tokenizer")
    parser.add_argument("--outdir",type=str,default="result",help="path to save result")

    parser.add_argument("--batch_size",type=int,default=32)
    parser.add_argument("--max_length",type=int,default=60)
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--lr",type=float,default=2e-5)
    args = parser.parse_args()
    return args

def train(dataloader, optimizer_, scheduler_, device_):
  # Use global variable for model.
  #global model

  # Tracking variables.
  predictions_labels = []
  true_labels = []
  total_loss = 0

  # Put the model into training mode.
  model.train()
  # For each batch of training data...
  for batch in tqdm(dataloader, total=len(dataloader)):

    # Add original labels - use later for evaluation.
    true_labels += batch['labels'].numpy().flatten().tolist()
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    model.zero_grad()

    outputs = model(**batch)

    loss, logits = outputs[:2]
    total_loss += loss.item()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer_.step()
    scheduler_.step()
    logits = logits.detach().cpu().numpy()
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()
  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)

  # Return all true labels and prediction for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss

def validation(model,dataloader, device_):
  # Use global variable for model.

  # Tracking variables
  predictions_labels = []
  true_labels = []
  #total loss for this epoch.
  total_loss = 0

  # Put the model in evaluation mode--the dropout layers behave differently
  # during evaluation.
  model.eval()

  # Evaluate data for one epoch
  for batch in tqdm(dataloader, total=len(dataloader)):

    # add original labels
    true_labels += batch['labels'].numpy().flatten().tolist()

    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}

    # Telling the model not to compute or store gradients, saving memory and
    # speeding up validation
    with torch.no_grad():
        outputs = model(**batch)
        loss, logits = outputs[:2]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        total_loss += loss.item()
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        predictions_labels += predict_content

  avg_epoch_loss = total_loss / len(dataloader)
  return true_labels, predictions_labels, avg_epoch_loss


def main():
    args = get_args()
    outdir  = args.outdir
    batch_size = args.batch_size
    max_length = args.max_length
    model_path =  args.pretrained_model
    epochs = args.epochs
    lr = args.lr

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    fout =open(outdir+"/metric_log.txt","w")
    print('Loading tokenizer...')
    tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.vocab)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("Loading dataset...")
    train_dataset = CustomDataset(args.train,tokenizer)
    n_labels = len(train_dataset.classes)
    labels_ids = train_dataset.labels_ids

    gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=labels_ids,
                                                          max_sequence_len=max_length)

    print('Created `train_dataset` with %d examples!'%len(train_dataset))

    # Move pytorch dataset into dataloader.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches!'%len(train_dataloader))

    valid_dataset = CustomDataset(args.valid,tokenizer)
    print('Created `valid_dataset` with %d examples!'%len(valid_dataset))

    # Move pytorch dataset into dataloader.
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

    print("Build model...")
    model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)
    model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)

    # resize model embedding to match new tokenizer
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id
    model.to(device)

    ##################
    optimizer = AdamW(model.parameters(),
                  lr = lr, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)

    # Store the average loss after each epoch so we can plot them.
    all_loss = {'train_loss':[], 'val_loss':[]}
    all_acc = {'train_acc':[], 'val_acc':[]}

    # Loop through each epoch.
    print('training...')
    for epoch in tqdm(range(epochs)):
        print()
        print('Training on batches...')
        train_predict = []
        train_labels = []
        total_loss = 0

        # Put the model into training mode.
        model.train()
        # For each batch of training data...
        i = 1
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            # Add original labels - use later for evaluation.
            epoch_labels = batch['labels'].numpy().flatten().tolist()
            train_labels += batch['labels'].numpy().flatten().tolist()
            batch = {k:v.type(torch.long).to(device) for k,v in batch.items()}
            model.zero_grad()

            outputs = model(**batch)
            loss, logits = outputs[:2]
            total_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            logits = logits.detach().cpu().numpy()
            train_predict += logits.argmax(axis=-1).flatten().tolist()
            #epoch_labels = batch['labels'].numpy().flatten().tolist()
            epoch_predict = logits.argmax(axis=-1).flatten().tolist()
            epoch_acc = accuracy_score(epoch_labels,epoch_predict)
            print("[Batch: {}/Epoch :{} ]  loss: {} - acc: {}".format(i,epoch, loss,epoch_acc))
            i+=1
        # Calculate the average loss over the training data.
        train_loss = total_loss / len(train_dataloader)
        # Perform one full pass over the training set.
        #train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
        train_acc = accuracy_score(train_labels, train_predict)

        # Get prediction form model on validation data.
        print('Validation on batches...')
        valid_labels, valid_predict, val_loss = validation(model,valid_dataloader, device)
        val_acc = accuracy_score(valid_labels, valid_predict)

        # Print loss and accuracy values to see how training evolves.
        fout.write("train_loss: %.5f,val_loss: %.5f,train_acc: %.5f,valid_acc: %.5f\n"%(train_loss, val_loss, train_acc, val_acc))
        print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
        print()

        # Store the loss value for plotting the learning curve.
        all_loss['train_loss'].append(train_loss)
        all_loss['val_loss'].append(val_loss)
        all_acc['train_acc'].append(train_acc)
        all_acc['val_acc'].append(val_acc)
        if (epoch+1)%10==0:
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(outdir + '/model_epoch_{}'.format(epoch + 1)):
                os.mkdir(outdir + '/model_epoch_{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(outdir + '/model_epoch_{}'.format(epoch + 1))
            # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
            # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
            print('epoch {} finished'.format(epoch + 1))

    
    true_labels, predictions_labels, avg_epoch_loss = validation(model,valid_dataloader, device)
    # Create the evaluation report.
    evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
    # Show the evaluation report.
    print(evaluation_report)

    print('training finished')
    if not os.path.exists(outdir + '/final_model'):
        os.mkdir(outdir + '/final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(outdir + '/final_model')
    torch.save(scheduler.state_dict(), outdir + '/final_model/scheduler.pt')
    torch.save(optimizer.state_dict(), outdir + '/final_model/optimizer.pt')

if __name__=="__main__":
    main()
