from pathlib import Path
import optuna, torch, numpy as np, os, shutil
import torch.optim as optim
from optuna.samplers import TPESampler
from optuna.visualization import plot_parallel_coordinate, plot_param_importances, plot_optimization_history, plot_edf

from config import BaseConfig, config
from utils.common import split_data, train_vae
from models.vae import VAE

def _suggest_basic_vae(trial: optuna.trial.Trial):
    base_cfg = BaseConfig()

    base_cfg.HIDDEN_DIM_1 = trial.suggest_categorical("HIDDEN_DIM_1", [512, 1024, 2048])
    base_cfg.HIDDEN_DIM_2 = trial.suggest_categorical("HIDDEN_DIM_2", [128, 256, 512])
    base_cfg.LATENT_DIM = trial.suggest_categorical("LATENT_DIM", [16, 32, 64])
    base_cfg.BETA_TYPE = trial.suggest_categorical("BETA_TYPE", ["fixed"])

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
    base_cfg.BETA_TYPE = trial.suggest_categorical("BETA_TYPE", ["fixed"])
    
    from models.encoders.conv_encoder import Encoder
    temp_encoder = Encoder(layer_params={
        "input_height":      base_cfg.INPUT_HEIGHT,
        "input_width":       base_cfg.INPUT_WIDTH,
        "intermediate_dims": [base_cfg.HIDDEN_DIM_1, base_cfg.HIDDEN_DIM_2, base_cfg.HIDDEN_DIM_3],
        "latent_dim":        base_cfg.LATENT_DIM
    })

    if temp_encoder.flattened_size > 50000:
        print(f"Pruning — flattened_size={temp_encoder.flattened_size}")
        raise optuna.exceptions.TrialPruned()

    return base_cfg

def _suggest_beta_vae(trial: optuna.trial.Trial):
    base_cfg = BaseConfig()

    channel_1 = trial.suggest_categorical("CHANNEL_1", [2, 4, 8, 16, 32])
    channel_2_multiplier = trial.suggest_categorical("CHANNEL_2_MULTIPLIER", [2, 4, 8])
    channel_3_multiplier = trial.suggest_categorical("CHANNEL_3_MULTIPLIER", [2, 4, 8])

    base_cfg.HIDDEN_DIM_3 = channel_1
    base_cfg.HIDDEN_DIM_2 = channel_1 * channel_2_multiplier
    base_cfg.HIDDEN_DIM_1 = channel_1 * channel_2_multiplier * channel_3_multiplier
    base_cfg.LATENT_DIM   = trial.suggest_categorical("LATENT_DIM", [16, 32, 64])
    base_cfg.BETA_TYPE = trial.suggest_categorical("BETA_TYPE", ["annealing"])
    base_cfg.BETA = trial.suggest_categorical("BETA", [1.0, 2.0, 3.0, 4.0, 5.0])
    
    from models.encoders.conv_encoder import Encoder
    temp_encoder = Encoder(layer_params={
        "input_height":      base_cfg.INPUT_HEIGHT,
        "input_width":       base_cfg.INPUT_WIDTH,
        "intermediate_dims": [base_cfg.HIDDEN_DIM_1, base_cfg.HIDDEN_DIM_2, base_cfg.HIDDEN_DIM_3],
        "latent_dim":        base_cfg.LATENT_DIM
    })

    if temp_encoder.flattened_size > 50000:
        print(f"Pruning — flattened_size={temp_encoder.flattened_size}")
        raise optuna.exceptions.TrialPruned()

    return base_cfg

def _suggest_cvae(trial: optuna.trial.Trial):
    base_cfg = BaseConfig()

    channel_1 = trial.suggest_categorical("CHANNEL_1", [2, 4, 8, 16, 32])
    channel_2_multiplier = trial.suggest_categorical("CHANNEL_2_MULTIPLIER", [2, 4, 8])
    channel_3_multiplier = trial.suggest_categorical("CHANNEL_3_MULTIPLIER", [2, 4, 8])

    base_cfg.HIDDEN_DIM_3 = channel_1
    base_cfg.HIDDEN_DIM_2 = channel_1 * channel_2_multiplier
    base_cfg.HIDDEN_DIM_1 = channel_1 * channel_2_multiplier * channel_3_multiplier
    base_cfg.LATENT_DIM   = trial.suggest_categorical("LATENT_DIM", [16, 32, 64])
    base_cfg.BETA_TYPE = trial.suggest_categorical("BETA_TYPE", ["fixed"])
    
    from models.encoders.conv_encoder import Encoder
    temp_encoder = Encoder(layer_params={
        "input_height":      base_cfg.INPUT_HEIGHT,
        "input_width":       base_cfg.INPUT_WIDTH,
        "intermediate_dims": [base_cfg.HIDDEN_DIM_1, base_cfg.HIDDEN_DIM_2, base_cfg.HIDDEN_DIM_3],
        "latent_dim":        base_cfg.LATENT_DIM
    })

    if temp_encoder.flattened_size > 50000:
        print(f"Pruning — flattened_size={temp_encoder.flattened_size}")
        raise optuna.exceptions.TrialPruned()

    return base_cfg

def _suggest_shared_parameters(trial: optuna.trial.Trial, base_cfg: BaseConfig):
    base_cfg.LR = trial.suggest_float("LR", 1e-4, 1e-3, log=True)
    base_cfg.BATCH_SIZE = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])
    
    return base_cfg

SEARCH_SPACES = {
    "basic": _suggest_basic_vae,
    "conv": _suggest_conv_vae,
    "beta": _suggest_beta_vae,
    "cvae": _suggest_cvae,
    "ae": _suggest_conv_vae
}

def make_objective_function(model_type, dataset, plot_dir_name=config.MODEL_TYPE, num_classes=0, device=None, epochs=config.EPOCHS, root=Path(".")):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if model_type not in SEARCH_SPACES:
        raise ValueError(f"Unknown model_type: `{model_type}`.\nAvailable: {list(SEARCH_SPACES.keys())}")
    
    def objective(trial: optuna.trial.Trial):
        trial_cfg = SEARCH_SPACES[model_type](trial=trial)
        trial_cfg = _suggest_shared_parameters(trial=trial, base_cfg=trial_cfg)
        
        print("\n"*3, "="*20, f"\nSUGGESTED PARAMS FOR TRIAL {trial.number}: {[item for item in trial_cfg.__dict__.items()]}\n", "="*20, "\n"*3)
        
        train_loader, test_loader = split_data(dataset=dataset, ratio=0.8, batch_size=trial_cfg.BATCH_SIZE, shuffle=config.SHUFFLE)
        
        model = VAE(cfg=trial_cfg, model_type=model_type, num_classes=num_classes).to(device=device)
        
        print("="*20, f"\n{model}\n", "="*20)
        
        optimizer = optim.Adam(model.parameters(), lr=trial_cfg.LR)
        
        history = train_vae(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            epochs=epochs, 
            annealing_epochs=trial_cfg.ANNEALING_EPOCHS,
            beta=trial_cfg.BETA,
            beta_type=trial_cfg.BETA_TYPE,
            device=device,
            trial_i=trial.number,
            plot_model_dir_name=plot_dir_name,
            root=root
        )
        
        last_test_recon = history["test_recon"][-1]
        
        if model_type == "ae":
            if np.isnan(last_test_recon):
                return float(1e6)
            return last_test_recon
        else:
            last_test_kl = history["test_kl"][-1]
            if np.isnan(last_test_recon) or np.isnan(last_test_kl):
                return float(1e6)
            return 0.6 * last_test_recon + 0.4 * last_test_kl
    
    return objective

def run_tuning(model_type, dataset, plot_dir_name=config.MODEL_TYPE, num_classes=0, device=None, epochs=config.EPOCHS, trials=config.TRIALS, root=Path(".")):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    # clearing trials plots dir for this study
    trial_plots_dir = root / config.RESULT_DIR / "trials" / plot_dir_name
    
    if trial_plots_dir.exists():
        print("Creating fresh trials directory...")
        shutil.rmtree(trial_plots_dir)
        
    study_name = {
        "basic": "Basic VAE",
        "conv":  "Convolutional VAE",
        "beta":  "Beta VAE",
        "cvae":  "Conditional VAE",
        "ae": "Convolutional AutoEncoder"
    }.get(model_type, model_type)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42, multivariate=True),
        study_name=f"{study_name} Tuning",
    )
    
    study.optimize(
        func=make_objective_function(
            model_type=model_type,
            dataset=dataset,
            plot_dir_name=plot_dir_name,
            num_classes=num_classes,
            device=device,
            epochs=epochs,
            root=root
        ),
        n_trials=trials,
        gc_after_trial=True,
        show_progress_bar=True
    )
    
    save_study_plots(study=study, plot_dir_name=plot_dir_name, root=root)
    
    return study
        
def save_study_plots(study: optuna.Study, plot_dir_name: str, root: Path=Path(".")):
    """Save informative Optuna study plots to results/trials/plots/<model_type>/"""
    plots_dir = root / config.RESULT_DIR / "trials" / plot_dir_name / "plots"
    plots_dir.mkdir(exist_ok=True, parents=True)
    
    print(plots_dir)

    plots = {
        "optimization_history":  plot_optimization_history(study),
        "param_importances":     plot_param_importances(study),
        "parallel_coordinate":   plot_parallel_coordinate(study),
        "edf":                   plot_edf(study),
    }

    for name, fig in plots.items():
        path = str(plots_dir / f"{name}.png")
        fig.write_image(path)
        print(f"Saved: {path}")