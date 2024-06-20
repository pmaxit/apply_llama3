import matplotlib.pyplot as plt
from transformers import TrainerCallback, TrainingArguments
from torch.optim import lr_scheduler

class LearningRateFinderCallback(TrainerCallback):
    def __init__(self,scheduler_type='exponential'):

        self.learning_rates = []
        self.losses = []
        self.scheduler_type = scheduler_type  # Will be initialized in on_init_end



    def on_log(self, args, state, control, logs=None, **kwargs):
        scheduler = kwargs['lr_scheduler']
        
        if logs is not None and "loss" in logs:
            lr = logs['learning_rate']
            loss = logs['loss']
            
            self.learning_rates.append(lr)
            self.losses.append(loss)
            if len(self.learning_rates) >= scheduler.total_iters:
                control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        # Plot learning rates and losses
        plt.figure(figsize=(10, 6))
        # Determine the size of the smallest list between lrs and losses
        min_size = min(len(self.learning_rates), len(self.losses))

        # Trim the lists to be of the same size
        self.learning_rates = self.learning_rates[:min_size]
        self.losses = self.losses[:min_size]

        fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,5))
        fig.suptitle('LR Finder')
        ax[0].plot(self.learning_rates)
        ax[1].plot(self.learning_rates, self.losses)
        ax[0].set_title('learning rate')
        ax[1].set_title('loss vs learning rate')

        print(self.learning_rates)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title("Learning Rate vs. Loss")
        plt.show()
        print("Learning rate finder complete. Please analyze the plot to select a suitable learning rate.")
