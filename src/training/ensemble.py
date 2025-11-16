import torch


class Ensemble:
    """Wrapper per ensemble di modelli"""

    def __init__(self, models):
        self.models = models
        for model in self.models:
            model.eval()

    def predict(self, x, device="cuda", batch_size=1024):
        """Predizione con batching per gestire grandi dataset"""
        n_samples = len(x)
        all_ensemble_preds = []

        # Process in batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            x_batch = x[start_idx:end_idx]

            batch_preds = []
            with torch.no_grad():
                for model in self.models:
                    pred = model(x_batch.to(device))
                    batch_preds.append(pred)

            # Media delle predizioni
            ensemble_pred = torch.stack(batch_preds).mean(dim=0)
            all_ensemble_preds.append(ensemble_pred.cpu())

        # Concatena tutti i batch
        final_pred = torch.cat(all_ensemble_preds, dim=0)

        return final_pred
