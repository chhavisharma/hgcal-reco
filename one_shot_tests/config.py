from os.path import expanduser
home = expanduser("~")

# DATA ROOT
data_root    = home+'/prototyping/data/train_1_/'

load_checkpoint_path = False

logfile_name = 'training.log'
logfile_name_test = 'testing.log'

total_epochs  = 500
train_samples = 500
batch_size    = 10 # this is a batch by hand so be careful
test_samples  = 100
input_classes = 50

plot_dir_root   = './plots/'
plot_dir_name   = 'train_event'+str(train_samples)+'_epoch'+str(total_epochs)+'_classes'+str(input_classes)
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

lr_param_gp_1     = 5e-2 #5e-3
lr_param_gp_2     = 5e-2 #0   
lr_param_gp_3     = 5e-2 #0 

reduceLR_factor   = 0.60
reduceLR_patience = 50