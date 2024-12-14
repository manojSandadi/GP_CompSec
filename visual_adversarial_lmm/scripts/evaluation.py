import os
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel

# Paths
ADV_DATASET_PATH = "adv_datasets/coco/classification"
COCO_IMAGE_PATH = "datasets/coco/val2014"
COCO_CAPTION_PATH = "datasets/coco/coco_2014val_caption.json"

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_captions(caption_path):
    """Load COCO captions."""
    with open(caption_path, 'r') as f:
        captions = json.load(f)
    return {item['image']: item['text'] for item in captions}

def create_labels_from_model(model, processor, image, text_labels):
    """Create labels based on the model's predictions."""
    model.eval()
    with torch.no_grad():
        if isinstance(image, torch.Tensor):
            # Denormalize tensor for CLIP compatibility
            image = image.permute(1, 2, 0).numpy()
            image = (image - image.min()) / (image.max() - image.min())
            image = Image.fromarray((image * 255).astype(np.uint8))
        inputs = processor(images=image, text=text_labels, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        predicted_label = logits_per_image.argmax(dim=1).item()
    return predicted_label

def visualize_images(original_img, adversarial_img, attack_name, perturbation, output_dir="visualizations"):
    """Visualize original and adversarial images side by side."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(adversarial_img)
    ax[1].set_title(f"{attack_name} ({perturbation})")
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{attack_name}_{perturbation}.png"))
    plt.close()

def evaluate_attack_performance(original_labels, adv_labels):
    """Evaluate performance of the attack."""
    accuracy = accuracy_score(original_labels, adv_labels)
    precision = precision_score(original_labels, adv_labels, average='weighted', zero_division=0)
    recall = recall_score(original_labels, adv_labels, average='weighted', zero_division=0)
    f1 = f1_score(original_labels, adv_labels, average='weighted', zero_division=0)

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}

def main():
    # Load captions and original labels
    captions = load_captions(COCO_CAPTION_PATH)
    text_labels = [captions[key][0] for key in captions.keys() if isinstance(captions[key], list) and captions[key]]  # Extract text labels for CLIP processing

    # Load pre-trained CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    attacks = os.listdir(ADV_DATASET_PATH)
    all_results = {}

    for attack in attacks:
        attack_path = os.path.join(ADV_DATASET_PATH, attack)
        perturbations = os.listdir(attack_path)

        for perturbation in perturbations:
            perturbation_path = os.path.join(attack_path, perturbation)
            attack_files = os.listdir(perturbation_path)

            original_labels = []
            adv_labels = []

            for adv_file in attack_files:
                file_path = os.path.join(perturbation_path, adv_file)

                # Ensure only .pt files are processed
                if not adv_file.endswith('.pt'):
                    print(f"Skipping non-.pt file: {adv_file}")
                    continue

                try:
                    adv_data = torch.load(file_path)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    continue

                # If labels are missing, generate from the model
                if isinstance(adv_data, torch.Tensor):
                    adv_img = adv_data
                else:
                    adv_img = adv_data.get('perturbed_image', None)

                if adv_img is None:
                    print(f"Missing data in file: {file_path}")
                    continue

                # Get original image
                img_name = adv_file.replace('.pt', '.jpg')
                original_img_path = os.path.join(COCO_IMAGE_PATH, img_name)
                if not os.path.exists(original_img_path):
                    continue

                original_img = Image.open(original_img_path).convert('RGB')

                # Generate labels using the model
                original_label = create_labels_from_model(model, processor, original_img, text_labels)
                adv_label = create_labels_from_model(model, processor, adv_img, text_labels)

                # Store labels for evaluation
                original_labels.append(original_label)
                adv_labels.append(adv_label)

                # Visualize original and adversarial images
                visualize_images(original_img, adv_img.permute(1, 2, 0).numpy(), attack, perturbation)

            # Evaluate attack performance
            if original_labels and adv_labels:
                results = evaluate_attack_performance(original_labels, adv_labels)
                all_results[f"{attack}_{perturbation}"] = results

    # Display results
    print("Attack Performance Summary:")
    for attack, metrics in all_results.items():
        print(f"\nAttack: {attack}")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.2f}")

    # Save table of results
    with open("evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
