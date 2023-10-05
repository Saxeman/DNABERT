# DNABERT

## DNABERT Example

### Python environment and requirements install
TODO: Environment installation and requirements.txt installation detailed instructions

## Training DNABERT with fine tune OVERVIEW

1. Download the tutorial train and test datasets csv files.
2. Download the DNABERT Pretrained model(DNABERT6 for this tutorial).
3. Edit the config.ini file.
5. Execute the main task.
6. Verify and observe training results and metrics.

# 1. Dataset

This example uses the Molecular Biology(Splice-junction Gene Sequences) Data Set [here](https://archive.ics.uci.edu/ml/datasets/Molecular+Biology+(Splice-junction+Gene+Sequences)). The task consists in creating a classification model for three classes(EI, IE, N), where N stands for neither.

The original dataset has the columns (**Class**, **Family**, and **Sequence**):

<table border = "1" class = "dataframe">
   <thead>
        <tr style = "text-align: right;">
            <th> </th>
            <th> family </th>
            <th > sequence </th >
            <th > class </th >
        </tr >
    </thead >
    <tbody >
       <tr >
            <th > 0</th>
            <td > ATRINS-DONOR-521</td>
            <td > CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG...</td>
            <td > EI</td>
        </tr >
        <tr >
            <th > 1</th>
            <td > ATRINS-DONOR-905</td>
            <td > AGACCCGCCGGGAGGCGGAGGACCTGCAGGGTGAGCCCCACCGCCCCTCCGTGCCCCCGC...</td>
            <td > EI</td>
        </tr >
        <tr >
            <th > 2</th>
            <td > BABAPOE-DONOR-30</td>
            <td > GAGGTGAAGGACGTCCTTCCCCAGGAGCCGGTGAGAAGCGCAGTCGGGGGCACGGGGATG...</td>
            <td > N</td>
        </tr >
        <tr >
            <th > 3</th>
            <td > BABAPOE-DONOR-867</td>
            <td > GGGCTGCGTTGCTGGTCACATTCCTGGCAGGTATGGGGCGGGGCTTGCTCGGTTTTCCCC...</td>
            <td > N</td>
        </tr >
        <tr >
            <th > 4</th>
            <td > BABAPOE-DONOR-2817</td>
            <td > GCTCAGCCCCCAGGTCACCCAGGAACTGACGTGAGTGTCCCCATCCCGGCCCTTGACCCT...</td>
            <td > IE</td>
        </tr >
    </tbody >
</table >
</div >

This implementation requires to reformat the dataset into the columns: `seq_id`, `seq_expr`, `seq_label`. In this example, the `seq_id` column is the < b >family</b> original column. As for `seq_expr`, the column <b>sequence</b> is renamed. Finally, column `seq_label` takes its values from the original <b>class</b> column and the encoding: `{'EI': 0, 'N': 1, 'IE': 2}`. The resulting dataset must look like this:

<table border = "1" class="dataframe">
   <thead >
        <tr style = "text-align: right;">
            <th > </th>
            <th > seq_id</th>
            <th > seq_expr</th>
            <th > seq_label</th>
        </tr >
    </thead >
    <tbody >
       <tr >
            <th > 0</th>
            <td > ATRINS-DONOR-521</td>
            <td > CCAGCTGCATCACAGGAGGCCAGCGAGCAGGTCTGTTCCAAGGGCCTTCGAGCCAGTCTG...</td>
            <td > 0</td>
        </tr >
        <tr >
            <th > 1</th>
            <td > ATRINS-DONOR-905</td>
            <td > AGACCCGCCGGGAGGCGGAGGACCTGCAGGGTGAGCCCCACCGCCCCTCCGTGCCCCCGC...</td>
            <td > 0</td>
        </tr >
        <tr >
            <th > 2</th>
            <td > BABAPOE-DONOR-30</td>
            <td > GAGGTGAAGGACGTCCTTCCCCAGGAGCCGGTGAGAAGCGCAGTCGGGGGCACGGGGATG...</td>
            <td > 1</td>
        </tr >
        <tr >
            <th > 3</th>
            <td > BABAPOE-DONOR-867</td>
            <td > GGGCTGCGTTGCTGGTCACATTCCTGGCAGGTATGGGGCGGGGCTTGCTCGGTTTTCCCC...</td>
            <td > 1</td>
        </tr >
        <tr >
            <th > 4</th>
            <td > BABAPOE-DONOR-2817</td>
            <td > GCTCAGCCCCCAGGTCACCCAGGAACTGACGTGAGTGTCCCCATCCCGGCCCTTGACCCT...</td>
            <td > 2</td>
        </tr >
    </tbody >
</table >
</div >

Processed data can be downloaded from here: [train_dataset](https://drive.google.com/file/d/1W1HK9iQLOf-ek3h3Kv5Mo8pWN6gMOyIj/view?usp=sharing), 
[test_dataset](https://drive.google.com/file/d/17JhEHKYQKnWJyLWbDEAXksQOPBlh-c4I/view?usp=sharing).

# 2. DNABERT Model

DNABERT Models were trained using kmer sequences. Therefore, the correct model must be used depending on the desired kmer value. Possible kmer values are 3, 4, 5, and 6. Please, dowload the correct model for your selected kmer(This example uses kmer=6) from:

[DNABERT3](https://drive.google.com/file/d/1nVBaIoiJpnwQxiz4dSq6Sv9kBKfXhZuM/view?usp=sharing)

[DNABERT4](https://drive.google.com/file/d/1V7CChcC6KgdJ7Gwdyn73OS6dZR_J-Lrs/view?usp=sharing)

[DNABERT5](https://drive.google.com/file/d/1KMqgXYCzrrYD1qxdyNWnmUYPtrhQqRBM/view?usp=sharing)

[DNABERT6](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)

After downloading the pretrained model, **unzip** it and keep track of the location path.

Using DNABERT involves a fine tuning approach, which means the complete body model(DNABERT) is trained alongside with the head model (Classification). In other words, the body and head weights are updated in training.

This implementation contains two versions of the DNABERT classification model.
> **Version A:** Simple classifier 

> **Version B:** Customizable classifier layer 

# Working Directory of The Tutorial
```
workdir
├── dataset_files           # Data directory of benchmark dataset
│   ├── train.csv           # Preprocesed train dataset
│   └── test.csv            # Preprocesed test dataset
│   └── val.csv             # Preprocesed val dataset
|
└── model
|   └── 6-new-12w-0         # Pretrained DNABERT6 Model
│
├── config.ini              # Configuration file
```

# 3. Main task

After activating the python virtual environent, execute

1. Run the main task with `python3 main.py --config <config.ini_path>`

```
[DATA_SOURCE]
train_csv_path = dataset_files/train.csv
val_csv_path = dataset_files/val.csv
test_csv_path = dataset_files/test.csv

[RESULTS_FILES]
results_path = results_files/

[MODEL]
model = DNABERT_B
dnabert_path = /home/ubuntu/Documents/MOLCURE/models/6-new-12w-0
kmer_val = 6
max_sequence_length = 79
mode = classification
head_layer_size = 128

[HYPERPARAMETERS]
batch_size = 32
test_batch_size = 32
max_epochs = 20
num_classes = 3
learning_rate = 0.0001
num_workers = 4
fine_tune = True
limit_train_batches = 1.0
accumulate_grad_batches = 2
log_every_n_steps = 10
deterministic = True
early_stop_monitor = val_loss
early_stop_mode = min
patience = 3
enable_checkpointing = False
```

Details for DNABERT specific options:
> [DATA_SOURCE]
1. `train_csv_path`: Train dataset csv file path
2. `val_csv_path`: Val dataset csv file path
3. `test_csv_path`: Test dataset csv file path
> [RESULTS_FILES]
1. `results_path`: Directory to store resulting files
> [MODEL]
1. `model`: DNABERT version. Options are [DNABERT_A], [DNABERT_B]
2. `dnabert_path`: DNABERT downloaded model directory path
3. `kmer_val`: Kmer value, **must** coincide with the downloaded DNABERT model
4. `max_sequence_length`: Max sequence length 
5. `head_layer_size`: Custom layer size. Only valid for DNABERT_B
> [HYPERPARAMETERS]
1. `batch_size`: Size of data to use in the mini batch training(Limited by memory)
2. `test_batch_size`: Size of data to use in the mini batch testing(Limited by memory)
3. `max_epochs`: Maximum number of epochs
4. `num_classes`: Number of categorical variables
5. `learning_rate`: Learning step size value
6. `num_workers`: Number of threads to handle data
7. `fine_tune`: Weather to use fine tuning or not. Required for the best DNABERT results
8. `limit_train_batches` = Percentage of dataset to use during training (1.0 means all of data) Used for debug
9. `accumulate_grad_batches`: Number of batches to wait for a grad optimization
10. `log_every_n_steps`: How many every n steps
11. `deterministic`: To replicate results
12. `early_stop_monitor`: What variable to monitor for early stopping. Validation loss
13. `early_stop_mode`: Min or max. Min is used for validation loss
14. `patience`: How many steps of grace before early stopping
15. `enable_checkpointing`: If you want to store the model after training using Lightning_logs