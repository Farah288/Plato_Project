from plato.algorithms import fedavg
# Import the base Server class
from plato.servers import fedavg as fedavg_server

class DRFedAvg(fedavg.Algorithm):
    """
    A customized FedAvg algorithm to correctly aggregate custom evaluation metrics 
    (F1, Recall, Precision) alongside standard metrics reported by the DRTrainer.
    """

    def aggregate_client_results(self, summaries, **kwargs):
        """
        Aggregates client testing results using a weighted average based on 
        the number of samples, specifically for custom metrics.
        """
        
        # 1. Call the base method for initial aggregation (Accuracy, Loss)
        # This also calculates the total_samples needed for weighted averaging.
        aggregated_results, total_samples = super().aggregate_client_results(
            summaries, **kwargs
        )
        
        if total_samples == 0:
            return aggregated_results, 0

        # --- 2. Custom Weighted Aggregation Logic for DR Metrics ---
        
        # Initialize custom metric totals
        f1_sum = 0.0
        recall_sum = 0.0
        precision_sum = 0.0
        
        # Iterate through all clients' reported summaries
        for summary in summaries:
            # The DRTestingStrategy stores all custom metrics under the 'test_results' key
            client_results = summary['results']['test_results']
            client_samples = client_results.get('samples', 0)
            
            # Calculate the weight for this client: (client's samples / total samples)
            weight = client_samples / total_samples

            # Apply the weight to the custom metrics
            f1_sum += client_results.get('f1_macro', 0.0) * weight
            recall_sum += client_results.get('recall_macro', 0.0) * weight
            precision_sum += client_results.get('precision_macro', 0.0) * weight

        # 3. Add the correctly aggregated custom metrics to the final results dictionary
        aggregated_results['f1_macro'] = f1_sum
        aggregated_results['recall_macro'] = recall_sum
        aggregated_results['precision_macro'] = precision_sum
        
        return aggregated_results, total_samples

class Server(fedavg_server.Server):
    """
    This class is the 'Server Wrapper'. It inherits the standard FedAvg Server 
    structure to satisfy the Plato Registry (it accepts model, algorithm, etc.).
    It uses your DRFedAvg class for its aggregation logic.
    """
    def __init__(self, model=None, algorithm=None, trainer=None):
        # 1. Create an instance of your custom Algorithm class
        custom_algorithm_instance = DRFedAvg(trainer=trainer)
        
        # 2. Pass your custom algorithm instance up to the base Server's __init__
        super().__init__(model=model, algorithm=custom_algorithm_instance, trainer=trainer)