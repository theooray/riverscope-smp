import os
import time, datetime
import argparse
import shutil

# =====================================================================
# Run several experiments in batch mode.
# =====================================================================
# Enables to run several experiments in batch mode.
# Each experiment runs the script train-test.py with a specific set of parameters.
# The user can define lists of hyperparameters, architectures, and encoders.
# ---------------------------------------------------------------------

# Argument parser.
parser = argparse.ArgumentParser()
parser.add_argument('--ds', type=str, default='deepglobe', help='Dataset name.')
parser.add_argument('--no_skip', action='store_true', help='Rerun experiments that already finished (by default they are skipped).')
parser.add_argument('--n_classes', type=int, default=None, help='Override the default number of classes of the dataset.')
args = parser.parse_args()

# Dataset-specific parameters.
h_size_dict = {'38-cloud' : 384,
               'deepglobe' : 512,
               'FUSAR-Map' : 512,
               'riverscope' : 512}
w_size_dict = {'38-cloud' : 384,
               'deepglobe' : 512,
               'FUSAR-Map' : 512,
               'riverscope' : 512}
in_channels_dict = {'38-cloud' : 4,
                    'deepglobe' : 3,
                    'FUSAR-Map' : 1,
                    'riverscope' : 4}
n_classes_dict = {'38-cloud' : 2,
                  'deepglobe' : 7,
                  'FUSAR-Map' : 5,
                  'riverscope' : 2}

# Number of classes: dataset default, unless overridden by --n_classes.
# A non-default number of classes saves the results in a separate folder (e.g., exp_riverscope_3classes),
# so they do not overwrite (or get skipped because of) the experiments with the default number of classes.
n_classes = args.n_classes if args.n_classes is not None else n_classes_dict[args.ds]
exp_tag = f'_{n_classes}classes' if n_classes != n_classes_dict[args.ds] else ''
exp_tag_str = f'--exp_tag {exp_tag} ' if exp_tag else ''

# Main experiment folder (must match EXP_PATH_MAIN in train-test.py).
EXP_PATH_MAIN = f'exp_{args.ds}{exp_tag}'

# Hyperparameters are defined as lists to run several experiments.
bs_list = [8] # [8, 16, 24]
lr_list = [0.0001] # [0.001, 0.0001, 0.00001]
loss_list = ['crossentropy', 'dice'] # ["jaccard", "dice", "tversky", "focal", "lavosz", "crossentropy"]
da_train_list = ['none', 'moderate'] # ['none', 'mild', 'moderate', 'strong']
scheduler_list = ['plateau'] # ['cosine', 'plateau', 'step']

# Architecture list
model_list_ = [
    'Unet', 
    'FPN',
    # Insert other models here.
]

# Encoder list
backbone_list_ = [
    'resnet50',            
    'efficientnet-b2',     
    # Insert other encoders here! 
]

max_epochs = 400 # 200, 400 # 1000
save_images_str = '--save_images' # --no-save_images
segmap_mode = 'darker' # ['simple', 'gray', 'darker']

# Inicia contagem de tempo deste época
time_start = time.time()

# Experiment counter.
ec = 0 

# Create the command string and run it.
for model in model_list_:
    for backbone in backbone_list_:
        for bs in bs_list:
            for lr in lr_list:
                for scheduler in scheduler_list:
                    for da_train in da_train_list:
                        for loss in loss_list:

                            # Skip experiments that already finished (e.g., when resuming after a crash/reboot).
                            # The folder name must match EXP_PATH in train-test.py, and the last report written
                            # by train-test.py at the end of the test evaluation is used as the "finished" marker.
                            exp_path = os.path.join(EXP_PATH_MAIN, f'exp_{model}_{backbone}_{loss}_{bs}_{lr}_{max_epochs}_{scheduler}_{da_train}')
                            done_file = os.path.join(exp_path, 'report_smp_(test)_(micro-imagewise).csv')
                            if not args.no_skip and os.path.exists(done_file):
                                print(f'Skipping experiment {ec} (already finished): {exp_path}')
                                ec = ec + 1
                                continue

                            ### for smp_reduction in smp_reduction_list:
                            cmd_str = f'nohup python train-test.py --dataset_name {args.ds} --n_classes {n_classes} {exp_tag_str}' + \
                                      f'--in_channels {in_channels_dict[args.ds]} --h_size {h_size_dict[args.ds]} --w_size {w_size_dict[args.ds]} ' + \
                                      f'--model {model} --backbone {backbone} --loss {loss} --da_train {da_train} --max_epochs {max_epochs} ' + \
                                      f'--batch_size {bs} --lr {lr} --scheduler {scheduler} {save_images_str} --segmap_mode {segmap_mode} ' + \
                                      f' --ec {ec} --resume' # --resume continues from last_checkpoint.pt if the experiment was interrupted.

                            ec = ec + 1

                            print(cmd_str)
                            os.system(cmd_str)

# Count total time of the experiment.
time_exp = time.time() - time_start
time_exp_hms = str(datetime.timedelta(seconds = time_exp))
print(f'Time exp.: {time_exp} sec ({time_exp_hms})')

print('\nFinish! (run-batch)')