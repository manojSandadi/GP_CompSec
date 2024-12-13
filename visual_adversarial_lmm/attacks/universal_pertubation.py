import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class UniversalAdversarialAttack:
    def __init__(self, model, epsilon=0.05, max_iter=100, target=None):
        """
        Initialize the universal adversarial attack.

        :param model: PyTorch model to attack.
        :param epsilon: Perturbation magnitude.
        :param max_iter: Maximum number of iterations for generating the attack.
        :param target: Target class for targeted attacks. If None, untargeted attack.
        """
        self.model = model
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.target = target
        self.device = next(model.parameters()).device

    def set_normalization_used(self, mean, std):
        """
        Set normalization statistics for preprocessing.
        :param mean: Mean for each channel.
        :param std: Standard deviation for each channel.
        """
        self.mean = torch.tensor(mean).view(1, -1, 1, 1).to(self.device)
        self.std = torch.tensor(std).view(1, -1, 1, 1).to(self.device)

    def generate(self, dataloader):
        """
        Generate the universal perturbation.

        :param dataloader: DataLoader containing the dataset.
        :return: Universal perturbation.
        """
        delta = torch.randn((1, *dataloader.dataset[0][0].shape), device=self.device) * 1e-6
        delta.requires_grad = True
        optimizer = optim.SGD([delta], lr=0.001)

        for _ in tqdm(range(self.max_iter), desc="Generating Universal Perturbation"):
            for batch_idx, batch in enumerate(dataloader):
                images, _, labels = batch[:3]
                images, labels = images.to(self.device), labels.to(self.device)

                print(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
                images = (images - self.mean) / self.std
                images = torch.clamp(images, -3, 3)

                perturbed_images = ((images + delta).clamp(0, 1) - self.mean) / self.std
                perturbed_images = torch.clamp(perturbed_images, -3, 3)

                print(f"Model input min: {perturbed_images.min()}, max: {perturbed_images.max()}, mean: {perturbed_images.mean()}")
                outputs = self.model(perturbed_images)

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    print(f"Model outputs contain NaN or Inf at batch {batch_idx + 1}, skipping...")
                    continue

                if self.target is not None:
                    loss = nn.CrossEntropyLoss()(outputs, torch.full_like(labels, self.target).to(self.device))
                else:
                    loss = -nn.CrossEntropyLoss()(outputs, labels)

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Loss is NaN or Inf at batch {batch_idx + 1}, skipping...")
                    continue

                print(f"Loss: {loss.item()}")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

        return delta.detach()





    def attack(self, images, delta):
        """
        Apply the universal perturbation to images.

        :param images: Input images.
        :param delta: Universal perturbation.
        :return: Perturbed images.
        """
        perturbed_images = (images + delta).clamp(0, 1)
        return perturbed_images

if __name__ == "__main__":
    from torchvision import datasets, transforms, models
    from data.dataloader import get_dataloader

    # Load pre-trained model
    model = models.resnet18(pretrained=True).eval().cuda()

    # Prepare dataset and dataloader
    class Args:
        data_file = "path/to/data_file.json"
        image_folder = "path/to/image_folder"
        task = "classification"
        model_path = "resnet18"
        image_ext = "jpg"
        batch_size = 32
        num_workers = 4

    args = Args()
    dataset, dataloader = get_dataloader(args, model)

    # Generate and apply universal adversarial attack
    attacker = UniversalAdversarialAttack(model, epsilon=0.05, max_iter=10, target=None)
    attacker.set_normalization_used(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    universal_delta = attacker.generate(dataloader)

    # Test the attack
    test_images, _, test_labels = next(iter(dataloader))
    perturbed_images = attacker.attack(test_images.cuda(), universal_delta)

    # Display results (Optional: requires matplotlib)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_images[i].permute(1, 2, 0).cpu().numpy())
        plt.title("Original")
        plt.axis("off")

        plt.subplot(2, 5, i + 6)
        plt.imshow(perturbed_images[i].permute(1, 2, 0).cpu().detach().numpy())
        plt.title("Perturbed")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
