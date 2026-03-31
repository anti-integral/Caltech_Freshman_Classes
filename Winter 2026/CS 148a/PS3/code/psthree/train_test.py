import torch
from tqdm import tqdm
import os

SEQUENCE_MODELS = ["rnn", "lstm", "transformer"]

def train(model, model_type, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, metrics_logger, results_path):

    print("\n")
    for epoch in range(num_epochs):
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Train", ascii=' >=')
        
        # TODO: Implement training loop
        for _, (inputs, targets) in enumerate(pbar):
            
            # zero the gradients of the optimizer
            # Hint: zero_grad() on optimizer
            

            # get outputs from the model with its forward pass
            # Hint: __call__ calls forward in a nn.Module
            outputs = None

            # If the model is a sequence model, get the last output
            # Don't worry about this till the RNN implementation onwards
            if model_type in SEQUENCE_MODELS:
                outputs = None

            # Compute the loss by calling the criterion
            loss = None

            # Backpropagate the loss
            

            # Update the model parameters using the optimizer's step

            # Convert output logits to probabilities using softmax
            # Use torch.nn.functional.softmax and specify the correct dimension
            probabilities = None
            
            # Get the classification predictions from output probabilities
            # Hint: Use torch.argmax on the appropriate dimension
            predictions = None
            
            # Update metrics logger and progress bar
            num_correct = (predictions == targets).sum().item()
            metrics_logger.update('train_loss', loss.item())
            metrics_logger.update('train_accuracy', 100*num_correct/targets.size(0))
            pbar.set_postfix({
                'loss': f'{metrics_logger.get_epoch_average("train_loss"):.4f}',
                'accuracy': f'{metrics_logger.get_epoch_average("train_accuracy"):.2f}%'
            })
        
        # TODO: Implement validation loop
        pbar = tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation", ascii=' >=')
        for inputs, targets in pbar:
            with torch.no_grad():

                # get outputs from the model with its forward pass
                outputs = None

                # If the model is a sequence model, get the last output
                # Don't worry about this till the RNN implementation onwards
                if model_type in SEQUENCE_MODELS:
                    outputs = None

                # Compute the loss by calling the criterion
                loss = None
                
                # Get the classification predictions from the softmaxed outputs
                predictions = None

                # Update metrics logger and progress bar
                num_correct = (predictions == targets).sum().item()
                metrics_logger.update('val_loss', loss.item())
                metrics_logger.update('val_accuracy', 100*num_correct/targets.size(0))
                pbar.set_postfix({
                    'loss': f'{metrics_logger.get_epoch_average("val_loss"):.4f}',
                    'accuracy': f'{metrics_logger.get_epoch_average("val_accuracy"):.2f}%'
                })
        
        metrics_logger.next_epoch()
        print("\n")

    os.makedirs(os.path.join(results_path, "plots"), exist_ok=True)
    
    metrics_logger.plot_metric(["train_loss", "val_loss"], 
                             os.path.join(results_path, "plots", "train_loss.png"))
    metrics_logger.plot_metric(["train_accuracy", "val_accuracy"], 
                             os.path.join(results_path, "plots", "train_accuracy.png"))
    
    # Save model and results
    print(f"Saving train results to {os.path.join(results_path, 'train_results.txt')}") 
    with open(os.path.join(results_path, "train_results.txt"), "w") as f:
        f.write(f"Train Loss: {metrics_logger.get_last_epoch_average('train_loss'):.4f}\n")
        f.write(f"Train Accuracy: {metrics_logger.get_last_epoch_average('train_accuracy'):.2f}%\n")
        f.write(f"Validation Loss: {metrics_logger.get_last_epoch_average('val_loss'):.4f}\n")
        f.write(f"Validation Accuracy: {metrics_logger.get_last_epoch_average('val_accuracy'):.2f}%\n")

    torch.save(model, os.path.join(results_path, f'{model_type}.pth'))
    print(f"Model saved at {os.path.join(results_path, f'{model_type}.pth')}")

def test(model, model_type, test_dataloader, criterion, metrics_logger, results_path):

    print("\n")
    pbar = tqdm(test_dataloader, desc=f"Test", ascii=' >=')

    # TODO: Implement test loop
    for _, (inputs, targets) in enumerate(pbar):
        with torch.no_grad():

            # get outputs from the model with its forward pass
            outputs = None

            # If the model is a sequence model, get the last output
            # Don't worry about this till the RNN implementation onwards
            if model_type in SEQUENCE_MODELS:
                outputs = None

            # Compute the loss by calling the criterion
            loss = None

            # Get the classification predictions from the softmaxed outputs
            predictions = None

            # Update metrics logger and progress bar
            num_correct = (predictions == targets).sum().item()
            metrics_logger.update('test_loss', loss.item())
            metrics_logger.update('test_accuracy', 100*num_correct/targets.size(0))

            pbar.set_postfix({'loss': f'{metrics_logger.get_epoch_average("test_loss"):.4f}', 
                                'accuracy': f'{metrics_logger.get_epoch_average("test_accuracy"):.2f}%'})
    print("\n")
    
    # Save results
    print(f"Saving test results to {os.path.join(results_path, 'test_results.txt')}")
    with open(os.path.join(results_path, 'test_results.txt'), "w") as f:
        f.write(f"Test Loss: {metrics_logger.get_epoch_average('test_loss'):.4f}\n")
        f.write(f"Test Accuracy: {metrics_logger.get_epoch_average('test_accuracy'):.2f}%\n")