from plato.trainers.composable import ComposableTrainer
# Import the custom strategy you just created in the same 'DR' package
from .dr_testing_strategy import DRTestingStrategy 

class DRTrainer(ComposableTrainer):
    """
    A custom trainer that inherits ComposableTrainer and injects 
    the DRTestingStrategy for custom evaluation metrics.
    """

    def __init__(self, model=None, callbacks=None):
        """Initialize trainer with the custom DRTestingStrategy."""
        
        # Instantiate your custom strategy
        dr_testing_strategy = DRTestingStrategy()

        # Call the parent ComposableTrainer's initializer
        # Inject ONLY the testing_strategy; keep all others as None to use defaults.
        super().__init__(
            model=model,
            callbacks=callbacks,
            loss_strategy=None,
            optimizer_strategy=None,
            training_step_strategy=None,
            lr_scheduler_strategy=None,
            model_update_strategy=None,
            data_loader_strategy=None,
            # THIS IS THE ONLY CHANGE: Injecting the custom strategy
            testing_strategy=dr_testing_strategy, 
        )
        
        # Retain the convenience attribute initialization from the base Trainer if needed
        self._loss_criterion = None

    # This fixes the model stagnation!
    def get_weights(self, model=None, **kwargs):
        """
        Ensures the final, locally trained model's state dictionary is 
        correctly returned to the client and sent to the server.
        """
        if model is None:
            model = self.model
            
        # Return the complete weight dictionary (state_dict) of the trained model
        return model.cpu().state_dict()