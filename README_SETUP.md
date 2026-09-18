# ¿Cómo configurar el entorno para bone_fracture_project?

Base propuesta para nuestro grupo: Python 3.12, PyTorch 2.8.0 y torchvision 0.23.0 para CPU. La opción GPU requiere un controlador NVIDIA funcional.

Estado local de Daniel: `.venv` tiene PyTorch `2.8.0+cu126` y torchvision `0.23.0+cu126`. La instalación, los imports y `pip check` se verificaron correctamente. El controlador funciona según `nvidia-smi` en su terminal. Daniel confirmó en su terminal un cálculo matricial y gradientes en la RTX 3050: GPU operativa con PyTorch. La prueba adicional de NMS en CUDA incluida abajo sigue disponible para verificar torchvision en GPU.

## 1. Requisitos del sistema

Ejecutar en una terminal:

```bash
python3 --version
python3 -m pip --version
git --version
df -h .
```

Solo si faltan los componentes, estos comandos instalan paquetes del sistema y requieren contraseña de administrador:

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git
```

## 2. Entrar al repositorio y crear el entorno

```bash
git clone https://github.com/sodanielstereo/bone_fracture_project.git
cd bone_fracture_project
```

Desde la raíz del repo, comprobar que `python3` es Python 3.12. Si `.venv` ya existe, inspeccionarlo antes de reutilizarlo; no sobrescribirlo sin revisar.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -c "import sys; print(sys.executable); assert sys.prefix != sys.base_prefix"
python -m pip --version
```

El ejecutable debe estar dentro de `bone_fracture_project/.venv/`. [Documentación de venv](https://docs.python.org/3.12/library/venv.html).

## 3. Dependencias

El archivo `requirements.txt` existente incluye torch, torchvision, numpy, matplotlib, pandas, Pillow, jupyter, ipykernel y scikit-learn. El paquete `jupyter` instala las interfaces necesarias. Se conserva como lista de dependencias directas.

Instalar PyTorch por separado permite escoger CPU o GPU. Como base explícita se propone el par compatible PyTorch 2.8.0 / torchvision 0.23.0, documentado oficialmente. [Versiones y comandos oficiales](https://pytorch.org/get-started/previous-versions/).

### CPU: base común del grupo

Con `.venv` activado y el archivo de dependencias preparado:

```bash
python -m pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r requirements-linux-py312-cpu.lock.txt
python -m pip check
```

El lock fija también las dependencias transitivas del entorno CPU.

### GPU NVIDIA: RTX 3050

En el equipo de Daniel, `nvidia-smi` ejecutado en la terminal de Linux Mint confirma la RTX 3050 de 6 GB con controlador 595.91.07 funcionando. No se necesita cambiar el controlador ni Secure Boot. La indicación CUDA 13.2 corresponde a la capacidad del controlador; PyTorch puede usar CUDA 12.6 con ese controlador más reciente. [Compatibilidad de NVIDIA](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

Escoger CPU o GPU para cada entorno. En el equipo de Daniel se reutiliza `.venv` reemplazando el par CPU por el par CUDA; no ejecutar luego la instalación CPU sobre ese entorno. Cerrar o reiniciar los kernels de Jupyter que ya estén abiertos después de cambiar PyTorch.

Para instalar GPU desde un entorno nuevo, o cambiar el entorno CPU existente:

```bash
source .venv/bin/activate
python -m pip install --no-cache-dir --upgrade 'torch==2.8.0+cu126' 'torchvision==0.23.0+cu126' --index-url https://download.pytorch.org/whl/cu126
python -m pip install --no-cache-dir -r requirements-linux-py312-cu126.lock.txt
python -m pip check
```

Se conserva el lock CPU para compañeros sin NVIDIA. El lock GPU registra las dependencias instaladas; la prueba de cálculo CUDA debe ejecutarse en una terminal con acceso a la GPU. No es necesario instalar el CUDA Toolkit del sistema para esta instalación de paquetes binarios.

## 4. Verificar imports y una operación real

Para verificar la GPU, ejecutar dentro de `.venv` en la terminal normal de Linux Mint:

```bash
python - <<'PY_GPU'
import torch, torchvision
from torchvision.ops import nms
print("PyTorch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA de PyTorch:", torch.version.cuda)
assert torch.cuda.is_available(), "CUDA no disponible en este proceso"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(256, 256, device="cuda", requires_grad=True)
loss = (x @ x.T).square().mean()
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all().item()
boxes = torch.tensor([[0., 0., 2., 2.], [0., 0., 2., 2.]], device="cuda")
scores = torch.tensor([0.9, 0.8], device="cuda")
assert nms(boxes, scores, 0.5).tolist() == [0]
torch.cuda.synchronize()
print("Cálculo, gradientes y torchvision en GPU: OK")
PY_GPU
```

En una instalación CPU, `torch.cuda.is_available()` será `False`: es lo esperado. La prueba anterior es específica para GPU. Un entorno aislado puede no tener acceso a los dispositivos NVIDIA aunque `nvidia-smi` funcione en la terminal normal.


## 5. Jupyter y kernel local

Registrar el kernel dentro del propio entorno, sin escribir en la configuración global del usuario:

```bash
python -m ipykernel install --sys-prefix --name bone-fracture --display-name "Python (bone_fracture_project)"
jupyter kernelspec list
jupyter lab
```

Seleccionar **Python (bone_fracture_project)**. En una celda, comprobar:

```python
import sys
print(sys.executable)
```

Debe apuntar al entorno elegido. Abrir Jupyter desde la terminal con ese entorno activado. Detener el servidor con Ctrl+C y salir del entorno con `deactivate`. [Documentación de kernels](https://ipython.readthedocs.io/en/stable/install/kernel_install.html).

## 6. Actualizar las versiones del grupo

Mantener `requirements.txt` como lista de dependencias directas. Tras aprobar los checks, guardar todas las versiones desde el entorno limpio CPU:

```bash
python -m pip freeze > requirements-linux-py312-cpu.lock.txt
```

Este comando crea o reemplaza el lock: revisar el diff antes de confirmarlo. Registrar también Python y pip utilizados en el README. El lock CPU del repositorio recoge el entorno validado. Para la primera instalación de cada integrante, usarlo en lugar de resolver de nuevo `requirements.txt`.

Para reconstruir CPU en otra máquina Linux con Python 3.12, crear y activar un entorno limpio, instalar primero el mismo par PyTorch CPU de la sección 3 y después:

```bash
python -m pip install -r requirements-linux-py312-cpu.lock.txt
python -m pip check
```

Repetir la prueba de la sección 4 y abrir un notebook. Para actualizar el lock desde el entorno GPU, usar `python -m pip freeze > requirements-linux-py312-cu126.lock.txt` después de comprobar las dependencias; registrar por separado el resultado de la prueba CUDA. Reconstruirlo instalando primero el par GPU correspondiente. No intercambiar locks CPU/GPU ni asumir compatibilidad con otros sistemas operativos. Fijar paquetes reproduce dependencias, pero no garantiza resultados de entrenamiento idénticos.

## 7. Estructura propuesta

```text
bone_fracture_project/
├── README.md
├── README_SETUP.md
├── requirements.txt
├── requirements-linux-py312-cpu.lock.txt   # versiones fijadas
├── requirements-linux-py312-cu126.lock.txt # entorno GPU
├── .gitignore
├── data/
│   ├── README.md
│   ├── mnist/                            # local
│   └── fracture_dataset/                 # local
├── notebooks/             # mnist/ y fractures/
├── src/
├── presentation/
├── reports/
└── .venv/                                # local
```

Conservar las carpetas que ya tenga el repo. Añadir al `.gitignore` existente, sin borrar sus reglas:

```gitignore
.venv/
.venv-*/
__pycache__/
*.py[cod]
.ipynb_checkpoints/
data/*
!data/README.md
checkpoints/
runs/
outputs/
*.pt
*.pth
*.ckpt
```

No descargar el dataset durante el setup del entorno. Documentamos en /data un README que contiene instrucciones para la extracción y estructura real en `data/README.md`.

Antes de un commit, comprobar `git status --short`: no deben aparecer el entorno, imágenes ni pesos. `.gitignore` no deja de seguir archivos que ya estuvieran versionados.

## Problemas frecuentes

- `externally-managed-environment`: activar el entorno y comprobar `sys.executable`; no forzar la instalación global.
- Error con `ensurepip` al crear `.venv`: instalar `python3-venv` para el Python del sistema y volver a crear un entorno en una ruta nueva.
- Notebook sin una librería instalada: comprobar el kernel y su `sys.executable`.
- Error de operadores de torchvision: comprobar que torch y torchvision forman el par acordado y que ambos usan CPU o la misma variante CUDA.
- Falta de espacio: revisar `df -h .`; no descargar simultáneamente dataset, entorno CPU y entorno GPU sin comprobar capacidad.
