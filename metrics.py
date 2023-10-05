import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassAUROC, MulticlassMatthewsCorrCoef, MulticlassConfusionMatrix
import torch
import matplotlib.pyplot as plt

class ClassificationMetrics:
    def __init__(self, num_classes) -> None:
        self.matt_score = MulticlassMatthewsCorrCoef(num_classes=num_classes)
        self.acc_score = MulticlassAccuracy(num_classes=num_classes)
        self.auroc_score = MulticlassAUROC(num_classes=num_classes)
        self.conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.softmax = torch.nn.Softmax(dim=0)
        self.num_classes = num_classes
        self.metrics_results = dict()
        
    def compute_metrics(self, predictions: np.array, labels: np.array, logits: np.array) -> dict:
        self.metrics_results["mcc_score"] = self.matt_score(torch.tensor(predictions), torch.tensor(labels))
        self.metrics_results["acc_score"] = self.acc_score(torch.tensor(predictions), torch.tensor(labels))
        self.metrics_results["auroc_score"] = self.auroc_score(torch.tensor(logits).float(), torch.tensor(labels).long())
        self.metrics_results["conf_matrix"] = self.conf_matrix(torch.tensor(predictions), torch.tensor(labels))
        return self.metrics_results
    
    def __compute_least_confidence(self, row: torch.tensor) -> torch.tensor:
        # Compute least confidence per logits row
        y_max = row.max()
        least_confidence = self.num_classes * (1 - y_max) / (self.num_classes - 1)
        return least_confidence
    
    def __compute_margin_confidence(self, row: torch.tensor) -> torch.tensor:
        # Compute margin confidence per logits row
        y_max, _ = torch.sort(row, descending=True)
        margin_confidence = 1 - (y_max[0] - y_max[1])
        return margin_confidence
    
    def __compute_ratio_confidence(self, row: torch.tensor) -> torch.tensor:
        # Compute ratio confidence per logits row
        y_max, _ = torch.sort(row, descending=True)
        ratio_confidence = y_max[1] / y_max[0]
        return ratio_confidence
    
    def __compute_entropy_confidence(self, row: torch.tensor) -> torch.tensor:
        # Compute entropy per logits row
        probs_log = row * torch.log2(row)
        entropy = 0 - torch.sum(probs_log) / torch.log2(torch.tensor(self.num_classes))
        return entropy
    
    def __compute_uncertainty_detail(self, logits: np.array) -> np.array:
        num_samples = logits.shape[0]
        least_confidence_array = np.zeros(num_samples)
        margin_confidence_array = np.zeros(num_samples)
        ratio_confidence_array = np.zeros(num_samples)
        entropy_array = np.zeros(num_samples)
        for idx in range(num_samples):
            row = logits[idx, :]
            row = self.softmax(torch.tensor(row, dtype=torch.double))
            least_confidence_array[idx] = self.__compute_least_confidence(row=row)
            margin_confidence_array[idx] = self.__compute_margin_confidence(row=row)
            ratio_confidence_array[idx] = self.__compute_ratio_confidence(row=row)
            entropy_array[idx] = self.__compute_entropy_confidence(row=row)
        
        metrics_concat = np.hstack((least_confidence_array.reshape(-1, 1), 
                                    margin_confidence_array.reshape(-1, 1),
                                    ratio_confidence_array.reshape(-1, 1),
                                    entropy_array.reshape(-1, 1))
                                    )
        
        return metrics_concat
    
    def compute_uncertainty_metrics(self, logits: np.array) -> pd.DataFrame:
        metrics_concat_array = self.__compute_uncertainty_detail(logits=logits)
        confidence_metrics_df = pd.DataFrame(metrics_concat_array, columns=["least_confidence", 
                                                                            "margin_confidence",
                                                                            "ratio_confidence",
                                                                            "entropy"])
        return confidence_metrics_df
    
    
# test_array = np.array([[1, 2, 3], [4, 6, 8]])
# metrics = ClassificationMetrics(num_classes=3)
# metrics_df = metrics.compute_uncertainty_metrics(logits=test_array)
# print(metrics_df.head())
