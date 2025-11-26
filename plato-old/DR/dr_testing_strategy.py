from plato.trainers.strategies.testing import DefaultTestingStrategy
from plato.config import Config
from sklearn.metrics import f1_score, recall_score, precision_score
import numpy as np
import torch

class DRTestingStrategy(DefaultTestingStrategy): 
    """
    Custom testing strategy for the DR dataset: Calculates F1, Recall, and 
    Precision (Macro-averaged) and stores them in the context for server aggregation.
    """

    def test_model(self, model, config, testset, sampler, context):
        """
        Runs the model on the test set and calculates all desired metrics.
        """
        
        # 1. Setup Environment and Data Loading
        model.to(context.device)
        model.eval()
        test_loader = self.create_test_loader(
            testset, sampler, Config().trainer.batch_size, context
        )
        
        all_true_labels = []
        all_predictions = []
        total_loss = 0.0
        
        # Get the loss function configured in the trainer
        loss_strategy = context.trainer.loss_strategy 

        # 2. Run Inference Loop
        with torch.no_grad():
            # Check for empty data set
            if test_loader is None:
                # Handle case where no test set is available for this client
                accuracy = 0.0
                num_samples = 0
                avg_loss = 0.0
                batch_count = 0
            else:
                for batch_id, (examples, labels) in enumerate(test_loader):
                    examples, labels = examples.to(context.device), labels.to(context.device)
                    
                    outputs = model(examples)
                    
                    # Calculate loss
                    loss = loss_strategy.compute_loss(outputs, labels, context)
                    total_loss += loss.item()
                    
                    # Get predicted class indices
                    _, predicted = torch.max(outputs.data, 1)
                    
                    # Collect batch results
                    all_true_labels.extend(labels.cpu().numpy())
                    all_predictions.extend(predicted.cpu().numpy())
                
                # Update final metrics after the loop
                y_true = np.array(all_true_labels)
                y_pred = np.array(all_predictions)
                num_samples = len(y_true)
                batch_count = batch_id + 1
                avg_loss = total_loss / batch_count
                accuracy = (y_true == y_pred).sum() / num_samples
        
        # 3. Final Metric Calculation (using scikit-learn metrics)
        try:
            # Use 'macro' average, which treats all classes equally, important for imbalance
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        except Exception:
            # Fallback for empty or problematic calculations
            f1, recall, precision = 0.0, 0.0, 0.0
            
        # 4. Store ALL results in the context
        context.state["test_results"] = {
            'loss': avg_loss,
            'accuracy': accuracy, 
            'f1_macro': f1,
            'recall_macro': recall,
            'precision_macro': precision,
            'samples': num_samples 
        }

        # The base framework expects the primary metric (accuracy) to be returned directly
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'samples': num_samples,
            # This key ('test_results') is what links the client's report to your 
            # custom aggregation logic (DRFedAvg) on the server.
            'test_results': context.state["test_results"] 
        }