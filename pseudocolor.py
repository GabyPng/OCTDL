import cv2
import numpy as np

REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


#------------------------------------------------------------ OpenCV

"""
Documentacion: https://docs.opencv.org/3.4.20/d3/d50/group__imgproc__colormap.html
"""
_CV_MAPS = {
    # frios
    "winter": cv2.COLORMAP_WINTER,
    "ocean": cv2.COLORMAP_OCEAN,
    "cool": cv2.COLORMAP_COOL,
    # calidos
    "hot": cv2.COLORMAP_HOT,
    "autumn": cv2.COLORMAP_AUTUMN,
    "inferno": cv2.COLORMAP_INFERNO,

    "jet": cv2.COLORMAP_JET,
    "viridis": cv2.COLORMAP_VIRIDIS,
}


def _make_cv(code):
    def fn(gray, cfg):
        return cv2.cvtColor(cv2.applyColorMap(gray, code), cv2.COLOR_BGR2RGB)
    return fn


for _name, _code in _CV_MAPS.items():
    REGISTRY[_name] = _make_cv(_code)


@register("none")
@register("gray3")
def _gray3(gray, cfg):
    """Baseline: gris replicado en 3 canales."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


# ------------------------------------------------------------ sin probar
DEFAULT_OCT_ANCHORS = [
    [0,   [0,   0,   40]],
    [50,  [0,   30,  120]],
    [90,  [0,   140, 200]],
    [140, [0,   220, 120]],
    [180, [230, 230, 0]],
    [220, [255, 120, 0]],
    [255, [255, 255, 255]],
]


def build_lut(anchors):
    """anchors: lista [[intensidad, [R, G, B]], ...] -> LUT 256x3 (RGB uint8)."""
    anchors = sorted(anchors, key=lambda a: a[0])
    xs = [a[0] for a in anchors]
    lut = np.zeros((256, 3), np.uint8)
    for c in range(3):
        ys = [a[1][c] for a in anchors]
        lut[:, c] = np.interp(np.arange(256), xs, ys).astype(np.uint8)
    return lut


@register("oct_custom")
def _oct_custom(gray, cfg):
    anchors = (cfg.get("oct_custom") or {}).get("anchors", DEFAULT_OCT_ANCHORS)
    return build_lut(anchors)[gray]        


# ------------------------------------------------------------ 
class PseudoColor:
    def __init__(self, cfg):
        cfg = cfg or {}
        self.name = str(cfg.get("name", "none")).lower()
        if self.name not in REGISTRY:
            raise ValueError(f"pseudocolor '{self.name}' no existe. Opciones: {sorted(REGISTRY)}")
        self.cfg = cfg
        pre = cfg.get("preprocess") or {}
        self.median_ksize = int(pre.get("median_ksize", 3))      # 0 
        self.clahe_clip = float(pre.get("clahe_clip", 2.0))      # 0 
        self.clahe_grid = int(pre.get("clahe_grid", 8))
        self._clahe = (cv2.createCLAHE(self.clahe_clip, (self.clahe_grid, self.clahe_grid))
                       if self.clahe_clip > 0 else None)

    def _to_gray(self, img):
        img = np.asarray(img)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if img.dtype != np.uint8:
            img = np.clip(img * (255 if img.max() <= 1.0 else 1), 0, 255).astype(np.uint8)
        return img

    def __call__(self, img):
        gray = self._to_gray(img)
        if self.median_ksize >= 3:
            gray = cv2.medianBlur(gray, self.median_ksize)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        return REGISTRY[self.name](gray, self.cfg)

    def __repr__(self):
        return f"PseudoColor(name={self.name})"


def build_pseudocolor(cfg):
    return PseudoColor(cfg)

# ------------------------------------------------------------ Generar dataset con filtro
"""
Copia --input_folder "OCTDL_procesado" a --output_folder "OCTDL_pseudocolor" aplicando el filtro.
La division es la misma que la del dataset de entrada.

Uso:
    python pseudocolor.py --name winter
    python pseudocolor.py --name hot --output_folder ./data/OCTDL_hot
    python pseudocolor.py --name gray3 --median_ksize 0 --clahe_clip 0    # gris original, sin procesar

Argumentos:
    --input_folder   (default ./data/OCTDL_procesado)
        Dataset ya dividido por preprocessing.py, con la estructura {train,val,test}/{normal,lesion}/.
        Las imagenes se leen en escala de grises.

    --output_folder  (default ./data/OCTDL_pseudocolor)
        Donde se guarda el dataset con filtro, con la misma estructura de carpetas y en PNG.

    --name           (default viridis)
        Filtro de color a aplicar:
            frios       winter, ocean, cool
            calidos     hot, autumn, inferno
            referencia  jet, viridis

    --median_ksize   (default 3)
        Tamano de la ventana del filtro de mediana que se aplica antes del color, para reducir el
        ruido speckle. Debe ser impar (3, 5, 7...); a mayor valor, mas suavizado. 0 = desactivado.

    --clahe_clip     (default 2.0)
        Limite de contraste de CLAHE, que se aplica despues de la mediana y antes del color.
        A mayor valor, mas contraste local (y mas ruido amplificado). 0 = desactivado.

    --clahe_grid     (default 8)
        CLAHE divide la imagen en una rejilla de clahe_grid x clahe_grid zonas y ecualiza cada una.
        A mayor valor, zonas mas pequenas y contraste mas local. Sin efecto si --clahe_clip es 0.
"""
if __name__ == "__main__":
    import argparse
    import shutil
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='./data/OCTDL_procesado', help='dataset ya dividido en train/val/test')
    parser.add_argument('--output_folder', type=str, default='./data/OCTDL_pseudocolor', help='dataset con filtro (se borra y se regenera)')
    parser.add_argument('--name', type=str, default='viridis', help='filtro: ' + ', '.join(sorted(REGISTRY)))
    parser.add_argument('--median_ksize', type=int, default=3, help='filtro de mediana antes del color (0 = desactivado)')
    parser.add_argument('--clahe_clip', type=float, default=2.0, help='contraste CLAHE (0 = desactivado)')
    parser.add_argument('--clahe_grid', type=int, default=8, help='rejilla de CLAHE')
    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    pseudocolor = build_pseudocolor({
        "name": args.name,
        "preprocess": {
            "median_ksize": args.median_ksize,
            "clahe_clip": args.clahe_clip,
            "clahe_grid": args.clahe_grid,
        },
    })
    print(pseudocolor, input_folder, '->', output_folder)

    if output_folder.exists():
        shutil.rmtree(output_folder)

    counts = {}
    for path in sorted(input_folder.rglob('*')):
        if not path.is_file():
            continue
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: No se pudo leer la imagen {path}.")
            continue
        # PNG (sin perdida): el JPEG submuestrea el color
        out_path = (output_folder / path.relative_to(input_folder)).with_suffix('.png')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(pseudocolor(gray), cv2.COLOR_RGB2BGR))
        key = str(out_path.parent.relative_to(output_folder))
        counts[key] = counts.get(key, 0) + 1

    for key, n in counts.items():
        print(key, n)
    print('total', sum(counts.values()))
