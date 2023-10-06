# Splice main
from MODELS.baseline import BaselineModelA, BaselineModelB
from dataset_handler import StandardDataset
from metrics import ClassificationMetrics
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
import pandas as pd
import numpy as np

def train_model(config_dict: dict):
    # Configuration variables
    data_source_config = config_dict["DATA_SOURCE"]
    model_config = config_dict["MODEL"]
    hyperparameters_config = config_dict["HYPERPARAMETERS"]
    results_config = config_dict["RESULTS_FILES"]
    
    # Hyperparameter variables
    BATCH_SIZE = int(hyperparameters_config["batch_size"])
    TEST_BATCH_SIZE = int(hyperparameters_config["test_batch_size"])
    MAX_EPOCHS = int(hyperparameters_config["max_epochs"])
    NUM_CLASSES = int(hyperparameters_config["num_classes"])
    num_workers = int(hyperparameters_config["num_workers"])
    learning_rate = float(hyperparameters_config["learning_rate"])
    fine_tune = True if hyperparameters_config["fine_tune"] == 'True' else False
    limit_train_batches = float(hyperparameters_config["limit_train_batches"])
    accumulate_grad_batches = int(hyperparameters_config["accumulate_grad_batches"])
    log_every_n_steps = int(hyperparameters_config["log_every_n_steps"])
    deterministic = True if hyperparameters_config["deterministic"] == 'True' else False
    early_stop_monitor = str(hyperparameters_config["early_stop_monitor"])
    patience = int(hyperparameters_config["patience"])
    early_stop_mode = str(hyperparameters_config["early_stop_mode"])
    enable_checkpointing = True if hyperparameters_config["enable_checkpointing"] == 'True' else False

    # Data source variables
    train_path = str(data_source_config["train_csv_path"])
    val_path = str(data_source_config["val_csv_path"])
    test_path = str(data_source_config["test_csv_path"])
    
    # Results variables
    results_path = str(results_config["results_path"])
    
    # Model variables
    model = str(model_config["model"])
    dnabert_path = str(model_config["dnabert_path"])
    max_sequence_length = int(model_config["max_sequence_length"])

    # Datasets and dataloaders
    train_ds = StandardDataset(path=train_path, 
                               max_length=max_sequence_length,
                               dnabert_path=dnabert_path)
    val_ds = StandardDataset(path=val_path, 
                             max_length=max_sequence_length,
                             dnabert_path=dnabert_path)
    test_ds = StandardDataset(path=test_path, 
                              max_length=max_sequence_length,
                              dnabert_path=dnabert_path)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    
    if model == "DNABERT_A":
        model = BaselineModelA(
            dnabert_path=dnabert_path,
            total_number_of_samples=len(train_ds),
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            fine_tune=fine_tune,
            lr=learning_rate,
            num_classes=NUM_CLASSES)
        
    elif model == "DNABERT_B":
        model = BaselineModelB(
            dnabert_path=dnabert_path,
            total_number_of_samples=len(train_ds),
            max_epochs=MAX_EPOCHS,
            batch_size=BATCH_SIZE,
            fine_tune=fine_tune,
            lr=learning_rate,
            num_classes=NUM_CLASSES)
    else:
        raise Exception(f"Model type error or not implemented! model = {model}")
    
    learning_rate_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(accelerator="gpu",
                        precision='16-mixed',
                        limit_train_batches=limit_train_batches,
                        accumulate_grad_batches=accumulate_grad_batches,
                        deterministic=deterministic, 
                        max_epochs=MAX_EPOCHS,
                        log_every_n_steps=log_every_n_steps,
                        enable_checkpointing=enable_checkpointing, 
                        callbacks=[EarlyStopping(monitor=early_stop_monitor, 
                                                 mode=early_stop_mode, 
                                                 patience=patience), 
                                    learning_rate_monitor])
    # Training
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)
    
    # Metrics computation
    metrics = ClassificationMetrics(num_classes=NUM_CLASSES)
    uncertainty_df = metrics.compute_uncertainty_metrics(logits=model.test_logits)
    print(uncertainty_df)
    metrics_df = metrics.compute_metrics(logits=model.test_logits, predictions=model.test_preds, labels=model.test_labels)
    print(metrics_df)
    
    # Result files
    results = np.hstack((model.test_preds.reshape(-1, 1), model.test_labels.reshape(-1, 1)))
    results_df = pd.DataFrame(results, columns=["Prediction", "True"])

    classes_list = range(0, NUM_CLASSES)
    logits_df = pd.DataFrame(model.test_logits, columns=classes_list)

    results_save_path = results_path + "dnabert_results.csv"
    logits_save_path = results_path + "dnabert_logits.csv"
    print(results_save_path)
    results_df.to_csv(results_save_path, index=False)
    logits_df.to_csv(logits_save_path, index=False)
    return
