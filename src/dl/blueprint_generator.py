"""
Deep Learning blueprint generation module for NeuroUrban system.
"""

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
import time

from src.config.settings import Config
from src.utils.gpu_utils import get_gpu_manager

class CityLayoutDataset(Dataset):
    """Dataset for city layout images."""

    def __init__(self, layouts: List[np.ndarray], transform=None):
        self.layouts = layouts
        self.transform = transform

    def __len__(self):
        return len(self.layouts)

    def __getitem__(self, idx):
        layout = self.layouts[idx]

        if self.transform:
            layout = self.transform(layout)

        return layout

class Generator(nn.Module):
    """GAN Generator for city layout generation."""

    def __init__(self, latent_dim: int = 100, img_size: int = 256):
        super(Generator, self).__init__()
        self.img_size = img_size

        # Calculate initial size
        init_size = img_size // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        self.init_size = init_size

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    """GAN Discriminator for city layout discrimination."""

    def __init__(self, img_size: int = 256):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(3, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # Calculate the size of the flattened features
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity

class BlueprintGenerator:
    """Generates city blueprints using deep learning."""

    def __init__(self, config: Config):
        """
        Initialize the blueprint generator.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager(config)
        self.device = self.gpu_manager.get_device()
        self.logger.info(f"Using device: {self.device}")

        # Mixed precision training
        self.use_mixed_precision = self.gpu_manager.mixed_precision_enabled
        self.scaler = GradScaler() if self.use_mixed_precision else None

        # Models
        self.generator = None
        self.discriminator = None

        # Training data
        self.training_data = None

        # Generated blueprints
        self.generated_blueprints = []

        # Performance tracking
        self.training_stats = {
            'epoch_times': [],
            'gpu_memory_usage': [],
            'loss_history': {'g_loss': [], 'd_loss': []}
        }

    def generate_blueprint(self, city_features: Optional[Dict] = None) -> str:
        """
        Generate a city blueprint.

        Args:
            city_features: Optional city features to condition generation

        Returns:
            Path to generated blueprint
        """
        self.logger.info("Generating city blueprint...")

        # Initialize models if not already done
        if self.generator is None:
            self._initialize_models()

        # Load or train models
        if not self._models_exist():
            self.logger.info("Training models first...")
            self._prepare_training_data()
            self._train_gan()
        else:
            self._load_models()

        # Generate blueprint
        blueprint = self._generate_single_blueprint(city_features)

        # Save blueprint
        blueprint_path = self._save_blueprint(blueprint)

        self.logger.info(f"✅ Blueprint generated and saved to: {blueprint_path}")
        return str(blueprint_path)

    def _initialize_models(self):
        """Initialize GAN models with GPU optimization."""
        self.logger.info("Initializing GAN models...")

        latent_dim = self.config.model.gan_latent_dim
        img_size = self.config.model.gan_image_size

        # Create models
        self.generator = Generator(latent_dim, img_size)
        self.discriminator = Discriminator(img_size)

        # Optimize models for current hardware
        self.generator = self.gpu_manager.optimize_model(self.generator)
        self.discriminator = self.gpu_manager.optimize_model(self.discriminator)

        # Initialize weights
        self._initialize_weights(self.generator)
        self._initialize_weights(self.discriminator)

        # Log model info
        self._log_model_info()

        self.logger.info("✅ Models initialized and optimized successfully")

    def _initialize_weights(self, model):
        """Initialize model weights using Xavier/He initialization."""
        for module in model.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight.data)
                if module.bias is not None:
                    nn.init.constant_(module.bias.data, 0)

    def _log_model_info(self):
        """Log model architecture and parameter information."""
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        gen_params = count_parameters(self.generator)
        disc_params = count_parameters(self.discriminator)

        self.logger.info(f"📊 Model Information:")
        self.logger.info(f"  Generator parameters: {gen_params:,}")
        self.logger.info(f"  Discriminator parameters: {disc_params:,}")
        self.logger.info(f"  Total parameters: {gen_params + disc_params:,}")
        self.logger.info(f"  Mixed precision: {self.use_mixed_precision}")

        # Log memory usage
        memory_info = self.gpu_manager.get_memory_info()
        if memory_info['device'] != 'cpu':
            self.logger.info(f"  GPU memory used: {memory_info['memory_used']:.2f} GB")

    def _models_exist(self) -> bool:
        """Check if trained models exist."""
        generator_path = self.config.get_model_path("generator.pth")
        discriminator_path = self.config.get_model_path("discriminator.pth")

        return generator_path.exists() and discriminator_path.exists()

    def _prepare_training_data(self):
        """Prepare training data for GAN."""
        self.logger.info("Preparing training data...")

        # Generate synthetic city layouts for training
        # In a real implementation, this would load actual city satellite images
        layouts = []

        for i in range(1000):  # Generate 1000 synthetic layouts
            layout = self._create_synthetic_layout()
            layouts.append(layout)

        # Create dataset with GPU-optimized transforms
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.config.model.gan_image_size, self.config.model.gan_image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.training_data = CityLayoutDataset(layouts, transform)
        self.logger.info(f"Prepared {len(layouts)} training samples")

        # Optimize batch size for current GPU
        if self.gpu_manager.is_gpu_available():
            optimal_batch_size = self.gpu_manager.get_optimal_batch_size(
                self.generator,
                (self.config.model.gan_latent_dim,),
                self.config.model.gan_batch_size
            )
            self.config.model.gan_batch_size = optimal_batch_size

    def _create_synthetic_layout(self) -> np.ndarray:
        """Create a synthetic city layout for training."""
        size = 256
        layout = np.zeros((size, size, 3), dtype=np.uint8)

        # Create PIL image for drawing
        img = Image.fromarray(layout)
        draw = ImageDraw.Draw(img)

        # Add different zones with different colors
        # Residential (green)
        for _ in range(np.random.randint(5, 15)):
            x1, y1 = np.random.randint(0, size-50, 2)
            x2, y2 = x1 + np.random.randint(20, 50), y1 + np.random.randint(20, 50)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 150, 0))

        # Commercial (blue)
        for _ in range(np.random.randint(3, 8)):
            x1, y1 = np.random.randint(0, size-40, 2)
            x2, y2 = x1 + np.random.randint(15, 40), y1 + np.random.randint(15, 40)
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 150))

        # Industrial (red)
        for _ in range(np.random.randint(2, 5)):
            x1, y1 = np.random.randint(0, size-60, 2)
            x2, y2 = x1 + np.random.randint(30, 60), y1 + np.random.randint(30, 60)
            draw.rectangle([x1, y1, x2, y2], fill=(150, 0, 0))

        # Roads (gray)
        for _ in range(np.random.randint(10, 20)):
            if np.random.random() > 0.5:  # Horizontal road
                y = np.random.randint(0, size)
                draw.rectangle([0, y, size, y+5], fill=(100, 100, 100))
            else:  # Vertical road
                x = np.random.randint(0, size)
                draw.rectangle([x, 0, x+5, size], fill=(100, 100, 100))

        # Parks (light green)
        for _ in range(np.random.randint(2, 6)):
            x, y = np.random.randint(0, size-30, 2)
            r = np.random.randint(10, 30)
            draw.ellipse([x, y, x+r, y+r], fill=(100, 200, 100))

        return np.array(img)

    def _train_gan(self):
        """Train the GAN model with GPU acceleration and mixed precision."""
        self.logger.info("🚀 Starting GPU-accelerated GAN training...")

        # Training parameters
        batch_size = self.config.model.gan_batch_size
        epochs = self.config.model.gan_epochs
        lr = self.config.model.gan_learning_rate

        # Create optimized data loader
        dataloader = self.gpu_manager.create_dataloader(
            self.training_data,
            batch_size=batch_size,
            shuffle=True
        )

        # Loss function
        adversarial_loss = nn.BCELoss()

        # Optimizers with improved settings
        optimizer_G = optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(self.config.model.gan_beta1, self.config.model.gan_beta2)
        )
        optimizer_D = optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(self.config.model.gan_beta1, self.config.model.gan_beta2)
        )

        # Learning rate schedulers
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

        # Training loop with mixed precision and performance monitoring
        self.logger.info(f"📊 Training Configuration:")
        self.logger.info(f"  Epochs: {epochs}")
        self.logger.info(f"  Batch size: {batch_size}")
        self.logger.info(f"  Learning rate: {lr}")
        self.logger.info(f"  Mixed precision: {self.use_mixed_precision}")

        for epoch in range(epochs):
            epoch_start_time = time.time()
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0

            for i, imgs in enumerate(dataloader):
                # Configure input
                real_imgs = imgs.to(self.device, non_blocking=True)
                current_batch_size = real_imgs.shape[0]

                # Adversarial ground truths
                valid = torch.ones(current_batch_size, 1, device=self.device, dtype=torch.float32)
                fake = torch.zeros(current_batch_size, 1, device=self.device, dtype=torch.float32)

                # Train Generator
                optimizer_G.zero_grad()

                # Sample noise
                z = torch.randn(current_batch_size, self.config.model.gan_latent_dim, device=self.device)

                if self.use_mixed_precision:
                    # Mixed precision training
                    with autocast():
                        gen_imgs = self.generator(z)
                        g_loss = adversarial_loss(self.discriminator(gen_imgs), valid)

                    self.scaler.scale(g_loss).backward()
                    self.scaler.step(optimizer_G)
                    self.scaler.update()
                else:
                    # Standard precision training
                    gen_imgs = self.generator(z)
                    g_loss = adversarial_loss(self.discriminator(gen_imgs), valid)
                    g_loss.backward()
                    optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()

                if self.use_mixed_precision:
                    with autocast():
                        real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
                        fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                        d_loss = (real_loss + fake_loss) / 2

                    self.scaler.scale(d_loss).backward()
                    self.scaler.step(optimizer_D)
                    self.scaler.update()
                else:
                    real_loss = adversarial_loss(self.discriminator(real_imgs), valid)
                    fake_loss = adversarial_loss(self.discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    d_loss.backward()
                    optimizer_D.step()

                # Accumulate losses
                epoch_g_loss += g_loss.item()
                epoch_d_loss += d_loss.item()

                # Log progress
                if i % 50 == 0:
                    memory_info = self.gpu_manager.get_memory_info()
                    if memory_info['device'] != 'cpu':
                        memory_str = f"GPU: {memory_info['memory_used']:.1f}GB"
                    else:
                        memory_str = "CPU"

                    self.logger.info(
                        f"Epoch [{epoch+1}/{epochs}] Batch [{i}/{len(dataloader)}] "
                        f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                        f"Memory: {memory_str}"
                    )

            # Update learning rates
            scheduler_G.step()
            scheduler_D.step()

            # Calculate epoch statistics
            epoch_time = time.time() - epoch_start_time
            avg_g_loss = epoch_g_loss / len(dataloader)
            avg_d_loss = epoch_d_loss / len(dataloader)

            # Store training statistics
            self.training_stats['epoch_times'].append(epoch_time)
            self.training_stats['loss_history']['g_loss'].append(avg_g_loss)
            self.training_stats['loss_history']['d_loss'].append(avg_d_loss)

            if self.gpu_manager.is_gpu_available():
                memory_info = self.gpu_manager.get_memory_info()
                self.training_stats['gpu_memory_usage'].append(memory_info['memory_used'])

            # Log epoch summary
            self.logger.info(
                f"✅ Epoch {epoch+1}/{epochs} completed in {epoch_time:.1f}s - "
                f"Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}"
            )

            # Clear cache periodically
            if (epoch + 1) % 10 == 0:
                self.gpu_manager.clear_cache()

        # Save trained models
        self._save_models()
        self._save_training_stats()
        self.logger.info("✅ GPU-accelerated GAN training completed successfully!")

    def _save_models(self):
        """Save trained models."""
        generator_path = self.config.get_model_path("generator.pth")
        discriminator_path = self.config.get_model_path("discriminator.pth")

        torch.save(self.generator.state_dict(), generator_path)
        torch.save(self.discriminator.state_dict(), discriminator_path)

        self.logger.info("Models saved successfully")

    def _save_training_stats(self):
        """Save training statistics."""
        stats_path = self.config.get_output_path("training_stats.json")

        # Calculate summary statistics
        if self.training_stats['epoch_times']:
            avg_epoch_time = np.mean(self.training_stats['epoch_times'])
            total_training_time = sum(self.training_stats['epoch_times'])

            summary = {
                'training_summary': {
                    'total_epochs': len(self.training_stats['epoch_times']),
                    'total_training_time_seconds': total_training_time,
                    'average_epoch_time_seconds': avg_epoch_time,
                    'final_g_loss': self.training_stats['loss_history']['g_loss'][-1] if self.training_stats['loss_history']['g_loss'] else 0,
                    'final_d_loss': self.training_stats['loss_history']['d_loss'][-1] if self.training_stats['loss_history']['d_loss'] else 0,
                    'mixed_precision_used': self.use_mixed_precision,
                    'device_used': str(self.device)
                },
                'detailed_stats': self.training_stats
            }

            with open(stats_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)

            self.logger.info(f"📊 Training statistics saved to: {stats_path}")
            self.logger.info(f"⏱️ Total training time: {total_training_time/60:.1f} minutes")
            self.logger.info(f"⚡ Average epoch time: {avg_epoch_time:.1f} seconds")

    def _load_models(self):
        """Load trained models."""
        generator_path = self.config.get_model_path("generator.pth")
        discriminator_path = self.config.get_model_path("discriminator.pth")

        self.generator.load_state_dict(torch.load(generator_path, map_location=self.device))
        self.discriminator.load_state_dict(torch.load(discriminator_path, map_location=self.device))

        self.generator.eval()
        self.discriminator.eval()

        self.logger.info("Models loaded successfully")

    def _generate_single_blueprint(self, city_features: Optional[Dict] = None) -> np.ndarray:
        """Generate a single city blueprint."""
        self.generator.eval()

        with torch.no_grad():
            # Generate random noise
            z = torch.randn(1, self.config.model.gan_latent_dim, device=self.device)

            # Modify noise based on city features if provided
            if city_features:
                z = self._condition_noise(z, city_features)

            # Generate image
            gen_img = self.generator(z)

            # Convert to numpy array
            gen_img = gen_img.cpu().squeeze().permute(1, 2, 0).numpy()

            # Denormalize
            gen_img = (gen_img + 1) / 2.0
            gen_img = np.clip(gen_img, 0, 1)

            # Convert to uint8
            blueprint = (gen_img * 255).astype(np.uint8)

        return blueprint

    def _condition_noise(self, z: torch.Tensor, city_features: Dict) -> torch.Tensor:
        """Condition the noise based on city features."""
        # Simple conditioning - modify noise based on features
        # In a more sophisticated implementation, this would use conditional GANs

        if "sustainability_focus" in city_features:
            # Increase green areas for sustainable cities
            z[:, :10] *= 1.5

        if "tech_focus" in city_features:
            # Modify for tech-focused cities
            z[:, 10:20] *= 1.3

        if "density" in city_features:
            # Adjust based on desired density
            density = city_features["density"]
            z[:, 20:30] *= density

        return z

    def _save_blueprint(self, blueprint: np.ndarray) -> str:
        """Save generated blueprint."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"city_blueprint_{timestamp}.png"
        blueprint_path = self.config.get_output_path(filename)

        # Save as image
        img = Image.fromarray(blueprint)
        img.save(blueprint_path)

        # Also save metadata
        metadata = {
            "timestamp": timestamp,
            "model_config": {
                "latent_dim": self.config.model.gan_latent_dim,
                "image_size": self.config.model.gan_image_size
            },
            "blueprint_path": str(blueprint_path)
        }

        metadata_path = self.config.get_output_path(f"blueprint_metadata_{timestamp}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return str(blueprint_path)

    def generate_multiple_blueprints(self, count: int = 5, city_features: Optional[Dict] = None) -> List[str]:
        """Generate multiple city blueprints."""
        self.logger.info(f"Generating {count} city blueprints...")

        blueprints = []
        for i in range(count):
            blueprint_path = self.generate_blueprint(city_features)
            blueprints.append(blueprint_path)
            self.logger.info(f"Generated blueprint {i+1}/{count}")

        return blueprints
