import json
import os

import torch
import torch.nn as nn
from torch import optim

from model import MLPClassifier
from logger import Logger
from eval import evaluate, print_results, plot_results


class Solver(object):
    """
    Same role as the reference repo's Solver, but for a single supervised
    classifier instead of a GAN: one model, one optimizer, one loss.
    """
    
    def __init__(self, config, train_loader, val_loader, test_loader):
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.num_epochs = config.num_epochs
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.log_step = config.log_step
        self.model_path = config.model_path
        self.results_path = config.results_path
        self.seed = config.seed

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.criterion = nn.CrossEntropyLoss()
        self.logger = Logger(config.logs_path)

        self.build_model()

    def build_model(self):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.model = MLPClassifier(num_classes=self.config.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr,
                                    weight_decay=self.weight_decay)

        n_params = sum(p.numel() for p in self.model.parameters())
        print("\nDataset configuration:")
        print("  train examples:", len(self.train_loader.dataset))
        print("  val examples:  ", len(self.val_loader.dataset))
        print("  test examples: ", len(self.test_loader.dataset))
        print("\nModel: MLPClassifier, {} classes, {:,} parameters, device: {}\n".format(
            self.config.num_classes, n_params, self.device))

    def save_checkpoint(self, name, epoch, val_acc):
        path = os.path.join(self.model_path, name)
        torch.save({
            "model_state": self.model.state_dict(),
            "config": vars(self.config),   # test.py rebuilds the model and data from this
            "epoch": epoch,
            "val_acc": val_acc,
        }, path)
        return path

    def train(self):
        total_step = len(self.train_loader)
        best_val_acc = -1.0
        best_path = None
        history = []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, running_correct, running_n = 0.0, 0, 0

            for i, (x, y, _snr) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * y.size(0)
                running_correct += (logits.argmax(dim=1) == y).sum().item()
                running_n += y.size(0)

                if (i + 1) % self.log_step == 0:
                    print('Epoch [{0:d}/{1:d}], Step [{2:d}/{3:d}], loss: {4:.4f}'.format(
                        epoch + 1, self.num_epochs, i + 1, total_step, loss.item()))
                    self.logger.scalar_summary('train/step_loss', loss.item(),
                                               epoch * total_step + i + 1)

            # ------------------- end of epoch: validate -------------------
            train_loss = running_loss / running_n
            train_acc = running_correct / running_n
            val = evaluate(self.model, self.val_loader, self.device)
            val_acc = val["overall_acc"]

            print('==> Epoch {:d}: train loss {:.4f}, train acc {:.4f}, val acc {:.4f}'.format(
                epoch + 1, train_loss, train_acc, val_acc))
            self.logger.scalar_summary('train/loss', train_loss, epoch + 1)
            self.logger.scalar_summary('train/acc', train_acc, epoch + 1)
            self.logger.scalar_summary('val/acc', val_acc, epoch + 1)
            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "train_acc": train_acc, "val_acc": val_acc})

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = self.save_checkpoint("best.pth", epoch + 1, val_acc)
                print('    new best model saved ({:.4f})'.format(val_acc))

        self.save_checkpoint("last.pth", self.num_epochs, val_acc)
        with open(os.path.join(self.results_path, "history.json"), "w") as f:
            json.dump(history, f, indent=4)
        self.logger.close()

        # ------------------- final test on the best model -------------------
        print("\nTraining done. Testing best model (val acc {:.4f}) on held-out test set...".format(best_val_acc))
        self.model.load_state_dict(torch.load(best_path, map_location=self.device)["model_state"])
        self.test()

    def test(self):
        results = evaluate(self.model, self.test_loader, self.device)
        print_results(results)
        with open(os.path.join(self.results_path, "results.json"), "w") as f:
            json.dump(results, f, indent=4)
        plot_results(results, os.path.join(self.results_path, "accuracy_vs_snr.png"))
        print("Saved results to", self.results_path)
        return results