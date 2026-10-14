
"""
Uso desde src/OCTDL:
    python gradcam.py --weights ./run_6/final_weights.pt --out ./gradcam_opcional 

Genera en ./gradcam_out/ cuatro imagenes con cuadriculas:
    FP_normal_pred_lesion.png   normales que el modelo marco como lesion 
    FN_lesion_pred_normal.png   lesiones que el modelo dejo pasar como normales
    OK_normal.png               normales bien clasificadas 
    OK_lesion.png               lesiones bien clasificadas 
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """Grad-CAM: pondera los mapas de activacion de una capa
    convolucional por el gradiente medio de la clase objetivo."""

    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.acts = output
        output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.grads = grad

    def __call__(self, x, class_idx=None):
        """x: tensor (B,3,H,W). Devuelve (cams (B,H,W) en [0,1], logits (B,C))."""
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)   # asegura que haya grafo aunque haya capas congeladas
            self.model.zero_grad()
            logits = self.model(x)
            if class_idx is None:
                idx = logits.argmax(dim=1)
            else:
                idx = torch.full((x.shape[0],), int(class_idx), device=x.device, dtype=torch.long)
            logits.gather(1, idx.view(-1, 1)).sum().backward()

            weights = self.grads.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((weights * self.acts).sum(dim=1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            flat = cam.flatten(1)
            mn = flat.min(dim=1)[0][:, None, None]
            mx = flat.max(dim=1)[0][:, None, None]
            cam = (cam - mn) / (mx - mn + 1e-8)
        return cam.detach().cpu().numpy(), logits.detach()


def find_target_layer(model):
    """Ultimo bloque convolucional de un ResNet (layer4[-1])"""
    modules = dict(model.named_modules())
    keys = [k for k in modules if k.split('.')[-1] == 'layer4']
    if not keys:
        raise RuntimeError('No encontre "layer4" en el modelo; hay que indicar la capa a mano')
    return modules[keys[0]][-1]


@torch.no_grad()
def predict_all(model, dataset, device, batch_size=32):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, probs = [], []
    for x, _ in loader:
        out = model(x.to(device))
        p = torch.softmax(out, dim=1)
        preds.append(p.argmax(1).cpu())
        probs.append(p.cpu())
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()


def draw_grid(items, title, out_path, gradcam, dataset, device, input_size, lesion_idx):
    if not items:
        print(f'  {title}: no hay casos, no se genera imagen.')
        return
    n = len(items)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6), squeeze=False)
    for col, i in enumerate(items):
        path, target = dataset.samples[i]
        x, _ = dataset[i]
        cam, logits = gradcam(x.unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim=1)[0, lesion_idx].item()
        img = Image.open(path).convert('RGB').resize((input_size, input_size))

        axes[0, col].imshow(img)
        axes[0, col].set_title(f'{Path(path).stem}\np(lesion)={prob:.2f}', fontsize=7)
        axes[1, col].imshow(img)
        axes[1, col].imshow(cam[0], cmap='jet', alpha=0.45)
        for row in (0, 1):
            axes[row, col].axis('off')
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'  guardado: {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True, help='ruta a final_weights.pt o best_validation_weights.pt')
    ap.add_argument('--config', default='OCTDL', help='nombre del yaml en ./configs')
    ap.add_argument('--n', type=int, default=8, help='imagenes por cuadricula')
    ap.add_argument('--out', default='./gradcam_out')
    ap.add_argument('--seed', type=int, default=0, help='semilla para elegir los casos "bien clasificados"')
    args = ap.parse_args()

# ---------------------------------------------------------------------------------------------- Indp
    from hydra import compose, initialize
    from data.builder import generate_dataset
    from modules.builder import generate_model

    with initialize(version_base=None, config_path='configs'):
        cfg = compose(config_name=args.config)

    device = cfg.base.device
    cfg.train.checkpoint = args.weights
    model = generate_model(cfg).to(device).eval()
    _, test_dataset, _ = generate_dataset(cfg)     
    input_size = cfg.data.input_size

    classes = test_dataset.classes
    lesion_idx, normal_idx = classes.index('lesion'), classes.index('normal')
    print('Clases:', classes)

    preds, _ = predict_all(model, test_dataset, device)
    targets = np.array([t for _, t in test_dataset.samples])
    print(f'Test: {len(targets)} imagenes, accuracy simple {(preds == targets).mean():.3f}')

    groups = {
        'FP_normal_pred_lesion': np.where((targets == normal_idx) & (preds == lesion_idx))[0].tolist(),
        'FN_lesion_pred_normal': np.where((targets == lesion_idx) & (preds == normal_idx))[0].tolist(),
        'OK_normal': np.where((targets == normal_idx) & (preds == normal_idx))[0].tolist(),
        'OK_lesion': np.where((targets == lesion_idx) & (preds == lesion_idx))[0].tolist(),
    }
    print({k: len(v) for k, v in groups.items()})

    rng = random.Random(args.seed)
    gradcam = GradCAM(model, find_target_layer(model))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, idxs in groups.items():
        chosen = idxs if len(idxs) <= args.n else rng.sample(idxs, args.n)
        draw_grid(chosen, name, out_dir / f'{name}.png', gradcam, test_dataset, device, input_size, lesion_idx)


if __name__ == '__main__':
    main()