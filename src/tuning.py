from pathlib import Path
import optuna, torch, numpy as np
import torch.optim as optim
from optuna.samplers import TPESampler
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_optimization_history, plot_slice

from config import BaseConfig, config
from utils.common import split_data, train_vae
from models.vae import VAE

def _suggest_basic_vae(trial: optuna.trial.Trial):
    base_cfg = BaseConfig()

    base_cfg.HIDDEN_DIM_1 = trial.suggest_categorical("HIDDEN_DIM_1", [512, 1024, 2048])
    base_cfg.HIDDEN_DIM_2 = trial.suggest_categorical("HIDDEN_DIM_2", [128, 256, 512])
    base_cfg.LATENT_DIM = trial.suggest_categorical("LATENT_DIM", [26, 32, 64])

    return base_cfg

def _suggest_conv_vae(trial: optuna.trial.Trial):
    base_cfg = BaseConfig()

    channel_1 = trial.suggest_categorical("CHANNEL_1", [2, 4, 8, 16, 32])
    channel_2_multiplier = trial.suggest_categorical("CHANNEL_2_MULTIPLIER", [2, 4, 8])
    channel_3_multiplier = trial.suggest_categorical("CHANNEL_3_MULTIPLIER", [2, 4, 8])

    base_cfg.HIDDEN_DIM_3 = channel_1
    base_cfg.HIDDEN_DIM_2 = channel_1 * channel_2_multiplier
    base_cfg.HIDDEN_DIM_1 = channel_1 * channel_2_multiplier * channel_3_multiplier
    base_cfg.LATENT_DIM   = trial.suggest_categorical("LATENT_DIM", [16, 32, 64])

    from models.encoders.conv_encoder import Encoder
    temp_encoder = Encoder(layer_params={
        "input_height":      base_cfg.INPUT_HEIGHT,
        "input_width":       base_cfg.INPUT_WIDTH,
        "intermediate_dims": [base_cfg.HIDDEN_DIM_1, base_cfg.HIDDEN_DIM_2, base_cfg.HIDDEN_DIM_3],
        "latent_dim":        base_cfg.LATENT_DIM
    })

    if temp_encoder.flattened_size > 50000:
        raise optuna.exceptions.TrialPruned()

    return base_cfg

def _suggest_beta_vae(trial: optuna.trial.Trial):
    pass

def _suggest_cvae(trial: optuna.trial.Trial):
    pass

def _suggest_shared_parameters(trial: optuna.trial.Trial, base_cfg: BaseConfig):
    base_cfg.LR = trial.suggest_float("LR", 1e-4, 1e-3, log=True)
    base_cfg.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])
    # base_cfg.SHUFFLE = trial.suggest_categorical("shuffle", [True, False])
    
    return base_cfg

SEARCH_SPACES = {
    "basic": _suggest_basic_vae,
    "conv": _suggest_conv_vae,
    "beta": _suggest_beta_vae,
    "cvae": _suggest_cvae
}

def make_objective_function(model_type, dataset, device=None, epochs=config.EPOCHS):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Unknown model_type: `{model_type}`.\nAvailable: {list(SEARCH_SPACES.keys())}")
    
    def objective(trial: optuna.trial.Trial):
        trial_cfg = SEARCH_SPACES[model_type](trial=trial)
        trial_cfg = _suggest_shared_parameters(trial=trial, base_cfg=trial_cfg)
        
        train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=trial_cfg.BATCH_SIZE, shuffle=config.SHUFFLE)
        
        model = VAE(cfg=trial_cfg, model_type=model_type).to(device=device)
        
        optimizer = optim.Adam(model.parameters(), lr=trial_cfg.LR)
        
        history = train_vae(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=epochs, 
            beta=trial_cfg.BETA,
            device=device,
            trial_i=trial.number
        )
        
        last_test_recon = history["test_recon"][-1]
        last_test_kl = history["test_kl"][-1]
        last_total = history["test_total"][-1]
        
        if np.isnan(last_test_recon) or np.isnan(last_test_kl) or np.isnan(last_total):
            return float(1e6)
        
        # return 0.6 * last_test_recon + 0.4 * last_test_kl
        return last_total
    
    return objective

def run_tuning(model_type, dataset, device=None, epochs=config.EPOCHS, trials=config.TRIALS):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        study_name=f"{"Basic VAE" if model_type == "basic" else "Convolutional VAE" if model_type == "conv" else "Beta VAE" if model_type == "beta" else "Conditional VAE"} Tuning"
    )
    
    study.optimize(
        func=make_objective_function(
            model_type=model_type,
            dataset=dataset,
            device=device,
            epochs=epochs
        ),
        n_trials=trials,
        show_progress_bar=True
    )
    
    plot_parallel_coordinate(study=study).show()
    plot_param_importances(study=study).show()
    plot_optimization_history(study=study).show()
    plot_slice(study=study).show()
    
    return study
        
        