import time, os
import torch
from helper_evaluation import compute_accuracy


def train_model(model, num_epochs, train_loader,
                valid_loader, test_loader, optimizer,
                device, logging_interval=50,
                scheduler=None,
                scheduler_on='valid_acc',
                out_dir=None,
                fldr_name=None):

    start_time = time.time()
    minibatch_loss_list, train_acc_list, valid_acc_list = [], [], []
    
    if fldr_name is not None:
              out_filename = fldr_name + "_out.txt"
    else:
              out_filename = "unknown_out.txt"
    out_path = os.path.join(out_dir, out_filename)
    if os.path.exists(out_path):
      os.remove(out_path)
      print("File being overwritten:", str(out_path))
    
    for epoch in range(num_epochs):

        model.train()
        for batch_idx, (features, targets) in enumerate(train_loader):

            features = features.to(device)
            targets = targets.to(device)

            # ## FORWARD AND BACK PROP
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad()

            loss.backward()

            # ## UPDATE MODEL PARAMETERS
            optimizer.step()
            
            

            # ## LOGGING
            minibatch_loss_list.append(loss.item())
            if not batch_idx % logging_interval:
              print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                        f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                        f'| Loss: {loss:.4f}')
              with open(out_path, 'a') as fi:
                  print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                        f'| Batch {batch_idx:04d}/{len(train_loader):04d} '
                        f'| Loss: {loss:.4f}', file=fi)
                
          

        model.eval()
        with torch.no_grad():  # save memory during inference
            train_acc = compute_accuracy(model, train_loader, device=device)
            valid_acc = compute_accuracy(model, valid_loader, device=device)
            print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                    f'| Train: {train_acc :.2f}% '
                    f'| Validation: {valid_acc :.2f}%')
            with open(out_path, 'a') as fi:
              print(f'Epoch: {epoch+1:03d}/{num_epochs:03d} '
                    f'| Train: {train_acc :.2f}% '
                    f'| Validation: {valid_acc :.2f}%', file=fi)
            train_acc_list.append(train_acc.item())
            valid_acc_list.append(valid_acc.item())

        elapsed = (time.time() - start_time)/60
        print(f'Time elapsed: {elapsed:.2f} min')
        with open(out_path, 'a') as fi:
        	print(f'Time elapsed: {elapsed:.2f} min', file=fi)
        
        if scheduler is not None:

            if scheduler_on == 'valid_acc':
                scheduler.step(valid_acc_list[-1])
            elif scheduler_on == 'minibatch_loss':
                scheduler.step(minibatch_loss_list[-1])
            else:
                raise ValueError(f'Invalid `scheduler_on` choice.')
        

    elapsed = (time.time() - start_time)/60
    print(f'Total Training Time: {elapsed:.2f} min')
    with open(out_path, 'a') as fi:
    	print(f'Total Training Time: {elapsed:.2f} min', file=fi)

    test_acc = compute_accuracy(model, test_loader, device=device)
    print(f'Test accuracy {test_acc :.2f}%')
    with open(out_path, 'a') as fi:
    	print(f'Test accuracy {test_acc :.2f}%', file=fi)

    return minibatch_loss_list, train_acc_list, valid_acc_list
