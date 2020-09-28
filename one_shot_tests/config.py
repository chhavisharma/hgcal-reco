'''
==========================================
[CONFIG INSTRUCTIONS]

-> To train from scratch, set
load_checkpoint_path = False

-> To load checkpoint and resume training, set
load_checkpoint_path = <path_to_checkpoint>
epochs = <remaining number of epochs to train>

-> To run only tests, set
load_checkpoint_path = <path_to_checkpoint>
testing_mode = True
==========================================
'''

from os.path import expanduser
home = expanduser("~")

# DATA ROOT
data_root    = home+'/prototyping/data/trackml/'
volume_layer_ids = [
    [8, 2], [8, 4], [8, 6], [8, 8], # barrel pixels
    [7, 2], [7, 4], [7, 6], [7, 8], [7, 10], [7, 12], [7, 14],# minus pixel endcap
    [9, 2], [9, 4], [9, 6], [9, 8], [9, 10], [9, 12], [9, 14], # plus pixel endcap
]
full_dataset = False

load_checkpoint_path = './checkpoints/train_event1000_epoch500_classes400_delta20/event1000_classes400_epoch430_loss7.79386e-02_edgeAcc9.25365e-01_checkpoint.pt'
logfile_name = 'training.log'
testing_mode = False

total_epochs  = 70 # was 430 earlier
train_samples = 1000
batch_size    = 20 #this is a batch by hand so be careful | needs to be a factor of train size
test_samples  = 500
input_classes = 400
input_class_delta = 20  # sqrt(input_classes)

dir_prefix      = 'test_event'+str(test_samples) if testing_mode else 'train_event'+str(train_samples)
plot_dir_root   = './plots/'
plot_dir_name   = dir_prefix + '_epoch'+str(total_epochs)+'_classes'+str(input_classes)+'_delta'+str(input_class_delta)
plot_path       = plot_dir_root+plot_dir_name+'/'

# Save Checkpoints Path
checkpoint_dir  = './checkpoints/'
checkpoint_path = checkpoint_dir+plot_dir_name

## MODEL
# Embedding Dim
input_dim  = 3
hidden_dim = 32
interm_out = None
output_dim = 3

# Regressor and Classifier Output Dim
ncats_out  = 2
nprops_out = 3

# EdgeCat Settings
k             = 8 
conv_depth    = 3
edgecat_depth = 6  # TRY DEPTH==3,tried - kills edgenet's performance
make_plots   = True
make_test_plots = True


# Learning Rates
lr_threshold_1    = 1e-4 #5e-3
lr_threshold_2    = 7.5e-4 #1e-3

lr_param_gp_1     = 1e-3 #5e-3
lr_param_gp_2     = 1e-3 #0   
lr_param_gp_3     = 1e-3 #0 

schedLR           = False
reduceLR_factor   = .70
reduceLR_patience = 30
