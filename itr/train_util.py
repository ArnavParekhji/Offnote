import time
import torch
import numpy as np
from pathlib import Path
from transformers import WEIGHTS_NAME, CONFIG_NAME


def init_seed():
    seed_val = 42
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def save_model(model, output_dir):

    output_dir = Path(output_dir)
    # Step 1: Save a model, configuration and vocabulary that you have fine-tuned

    # If we have a distributed model, save only the encapsulated model
    # (it was wrapped in PyTorch DistributedDataParallel or DataParallel)
    model_to_save = model.module if hasattr(model, 'module') else model

    # If we save using the predefined names, we can load using `from_pretrained`
    output_model_file = output_dir / WEIGHTS_NAME
    output_config_file = output_dir / CONFIG_NAME

    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    #src_tokenizer.save_vocabulary(output_dir)

def load_model():
    pass

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):

    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    #print (f'preds: {pred_flat}')
    #print (f'labels: {labels_flat}')

    return np.sum(np.equal(pred_flat, labels_flat)) / len(labels_flat)

def run_train(config, model, train_loader, eval_loader, writer):
    init_seed()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader), eta_min=config.lr)
    
    training_loss_values = []
    validation_loss_values = []
    validation_accuracy_values = []

    for epoch in range(config.epochs):

        model.train()

        print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.epochs))
        start_time = time.time()

        total_loss = 0

        for batch_no, batch in enumerate(train_loader):
            print("Batch no: " + str(batch_no+1) + "/" + str(len(train_loader)))
            source = batch[0].to(device)
            target = batch[1].to(device)

            model.zero_grad()        

            loss, logits = model(source, target)
            total_loss += loss.item()
        
            logits = logits.detach().cpu().numpy()
            label_ids = target.to('cpu').numpy()

            loss.backward()

            optimizer.step()
            scheduler.step()

        #Logging the loss and accuracy (below) in Tensorboard
        avg_train_loss = total_loss / len(train_loader)            
        training_loss_values.append(avg_train_loss)

        for name, weights in model.named_parameters():
            writer.add_histogram(name, weights, epoch)

        writer.add_scalar('Train/Loss', avg_train_loss, epoch)

        print("Average training loss: {0:.2f}".format(avg_train_loss))
        print("Running Validation...")

        model.eval()

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps = 0

        for batch_no, batch in enumerate(eval_loader):
            print("Batch no eval: " + str(batch_no+1) + "/" + str(len(eval_loader)))
            source = batch[0].to(device)
            target = batch[1].to(device)
        
            with torch.no_grad():        
                loss, logits = model(source, target)

            logits = logits.detach().cpu().numpy()
            label_ids = target.to('cpu').numpy()
        
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy
            eval_loss += loss

            nb_eval_steps += 1

        avg_valid_acc = eval_accuracy/nb_eval_steps
        avg_valid_loss = eval_loss/nb_eval_steps
        validation_loss_values.append(avg_valid_loss)
        validation_accuracy_values.append(avg_valid_acc)

        writer.add_scalar('Valid/Loss', avg_valid_loss, epoch)
        writer.add_scalar('Valid/Accuracy', avg_valid_acc, epoch)
        writer.flush()

        print("Avg Val Accuracy: {0:.2f}".format(avg_valid_acc))
        print("Average Val Loss: {0:.2f}".format(avg_valid_loss))
        print("Time taken by epoch: {0:.2f}".format(time.time() - start_time))

    return training_loss_values, validation_loss_values, validation_accuracy_values


