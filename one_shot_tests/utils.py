import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from  mpl_toolkits import mplot3d

def logtofile(path, filename, logs):
    filepath = path + '/'+ filename
    if os.path.exists(filepath):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w'
    logfile = open(filepath,append_write)
    logfile.write(logs)
    logfile.write('\n')
    logfile.close()

def save_checkpoint(model_state, is_best, checkpoint_dir, checkpoint_name):
    f_path = os.path.join(checkpoint_dir,checkpoint_name + '_checkpoint.pt')
    torch.save(model_state, f_path)
    if is_best:
        best_fpath = os.path.join(checkpoint_dir, 'best_model_checkpoint.pt')
        shutil.copyfile(f_path, best_fpath)

def load_checkpoint(load_checkpoint_path, model, optimizer, scheduler):

    checkpoint = torch.load(config.load_checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['converged_categorizer'], \
                                    checkpoint['converged_embedding'], checkpoint['best_loss'] 


def plot_event(my_data,y_t):

    x,y,z = my_data[:,0], my_data[:,1], my_data[:,2]

    fig = plt.figure(figsize = (15, 10)) 
    ax1 = fig.add_subplot(111,projection='3d')
    
    #Axis 1 - hits 
    ax1.set_xlabel('Z-axis', fontweight ='bold')  
    ax1.set_ylabel('Y-axis', fontweight ='bold')  
    ax1.set_zlabel('X-axis', fontweight ='bold')  
    ax1.scatter3D(z, y, x, s=10, color= m.to_rgba(y_t), edgecolors='black')      

    ctr=0
    plt.savefig(config.plot_path+'event_'+str(ctr)+'.pdf') 

    plt.close(fig)