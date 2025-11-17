import torch

from tqdm import tqdm

N_SAMPLES = 5


class Ensemble:
    """Wrapper for ensemble of models"""

    def __init__(self, models):
        self.models = models
        for model in self.models:
            model.eval()

    def predict_single(self, model, x, device):
        """Prediction with Monte Carlo Dropout for uncertainty estimation"""

        def enable_dropout(m):
            """Function to enable the dropout layers during test-time"""
            if isinstance(m, torch.nn.Dropout):
                m.train()

        model.eval()
        model.apply(enable_dropout)

        all_predictions = []

        with torch.no_grad():
            for i in range(N_SAMPLES):
                pred = model(x.to(device))
                all_predictions.append(pred)

        return torch.stack(all_predictions).mean(dim=0)

    def predict(
        self,
        x,
        device,
        batch_size=1024,
    ):
        """Predict with ensemble of models using TTA with MC Dropout"""
        n_samples = len(x)
        all_model_preds = []

        for model in tqdm(self.models, desc="TTA for model"):
            model_preds = []

            # Process in batches
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                x_batch = x[start_idx:end_idx]
                pred = self.predict_single(model, x_batch, device)
                model_preds.append(pred.cpu())

            # Catenate batch predictions for the model
            model_pred_full = torch.cat(model_preds, dim=0)
            all_model_preds.append(model_pred_full)

        # Average predictions from all models
        return torch.stack(all_model_preds).mean(dim=0)
