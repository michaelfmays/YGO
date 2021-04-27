import torch, torchvision, os
from helper_funcs import set_all_seeds
from torchsampler import ImbalancedDatasetSampler
from helper_train import train_model

class settings():
  def __init__(self, RANDOM_SEED, BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE,
               current_dir, image_fldr, transform_list):
    self.RANDOM_SEED=RANDOM_SEED
    self.BATCH_SIZE=BATCH_SIZE
    self.NUM_EPOCHS=NUM_EPOCHS
    self.LEARNING_RATE=LEARNING_RATE
    
    self.out_dir = os.path.join(current_dir, "out")
    self.image_fldr = image_fldr
    
    self.TRAIN_DATA_PATH = os.path.join(current_dir, "sorted_cards",
                                        self.image_fldr, "train")
    self.TEST_DATA_PATH = os.path.join(current_dir, "sorted_cards",
                                       self.image_fldr, "test")
    self.VALID_DATA_PATH = os.path.join(current_dir, "sorted_cards",
                                        self.image_fldr, "valid")
    
    

    self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set number of classes to be number of folders in image directory
    os.chdir(self.TRAIN_DATA_PATH)
    stdout = os.popen('ls -l | grep "^d" | wc -l')
    self.NUM_GROUPS = int(str(stdout.read()).replace("\n", ""))
    os.chdir(current_dir)

    self.TRANSFORM_IMG = torchvision.transforms.Compose(transform_list)

  def generate_data_loaders(self):

    set_all_seeds(self.RANDOM_SEED)

    self.train_data = torchvision.datasets.ImageFolder(root=self.TRAIN_DATA_PATH, 
                                                  transform=self.TRANSFORM_IMG)
    self.train_loader = torch.utils.data.DataLoader(self.train_data, 
                                              sampler=ImbalancedDatasetSampler(self.train_data),
                                              batch_size=self.BATCH_SIZE, 
                                              #shuffle=True, 
                                              num_workers=2)

    self.valid_data = torchvision.datasets.ImageFolder(root=self.VALID_DATA_PATH, 
                                                  transform=self.TRANSFORM_IMG)
    self.valid_loader = torch.utils.data.DataLoader(self.valid_data, 
                                              sampler=ImbalancedDatasetSampler(self.valid_data),
                                              batch_size=self.BATCH_SIZE, 
                                              #shuffle=True, 
                                              num_workers=2)

    self.test_data = torchvision.datasets.ImageFolder(root=self.TEST_DATA_PATH, 
                                                transform=self.TRANSFORM_IMG)
    self.test_loader  = torch.utils.data.DataLoader(self.test_data, 
                                              sampler=ImbalancedDatasetSampler(self.test_data),
                                              batch_size=self.BATCH_SIZE, 
                                              #shuffle=True, 
                                              num_workers=2)

class model_output():
  def __init__(self, minibatch, train, valid):
    self.minibatch_loss_list = minibatch
    self.train_acc_list = train
    self.valid_acc_list = valid

class make_model():
  """
  Note: The optimizer and scheduler should both be passed as *methods* not *calls*
  		(no parentheses).
  """
  def __init__(self,
               model, settings,
               optimizer=None, optimizer_args=None,
               scheduler=None, scheduler_args=None, 
               scheduler_on="valid_acc",
               logging_interval=100):
    
    self.model = model.to(settings.DEVICE)
    self.settings = settings
    self.scheduler_on = scheduler_on
    self.logging_interval = logging_interval

    if optimizer is not None:
      if optimizer_args is not None:
        self.optimizer = optimizer(
                                   self.model.parameters(), 
                                   lr=settings.LEARNING_RATE,
                                   **optimizer_args
                                  )
      else:
        self.optimizer = optimizer(
                                   self.model.parameters(), 
                                   lr=settings.LEARNING_RATE
                                  )
    else:
      raise ValueError("Optimizer must not be None.")
        
    if scheduler is not None:
      if scheduler_args is not None:
        self.scheduler = scheduler(self.optimizer,
                                  **scheduler_args)
      else:
        self.scheduler = scheduler(self.optimizer)
  
  def run_train(self):
    
    minibatch_loss_list, train_acc_list, valid_acc_list = train_model(
      model=self.model,
      num_epochs=self.settings.NUM_EPOCHS,
      train_loader=self.settings.train_loader,
      valid_loader=self.settings.valid_loader,
      test_loader=self.settings.test_loader,
      optimizer=self.optimizer,
      device=self.settings.DEVICE,
      scheduler=self.scheduler,
      scheduler_on=self.scheduler_on,
      logging_interval=self.logging_interval,
      out_dir=self.settings.out_dir,
      fldr_name=self.settings.image_fldr)
    
    return model_output(minibatch_loss_list, train_acc_list, valid_acc_list)