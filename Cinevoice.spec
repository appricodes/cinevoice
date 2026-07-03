# -*- mode: python ; coding: utf-8 -*-
import os
import PySide6
import certifi
from PyInstaller.utils.hooks import copy_metadata, collect_dynamic_libs

# 1. Dynamically locate your PySide6 installation plugins directory
pyside6_dir = os.path.dirname(PySide6.__file__)
plugins_dir = os.path.join(pyside6_dir, 'plugins')

datas = []
datas += copy_metadata('keyring')
# FORCE PyInstaller to pack the SSL certificates for secure API requests
datas.append((certifi.where(), 'certifi'))
datas.append(('help.html', '.'))
datas.append(('LICENSE.txt', '.'))
datas.append(('THIRD-PARTY-LICENSES.txt', '.'))

# OpenVINO runtime for offline models: bundle the libs dir so device plugins
# (CPU/GPU) and their registration files (cache.json) stay together, plus the
# GenAI and tokenizer native libraries. NPU plugin DLLs (~90 MB) and static
# import .lib files are dropped; OpenVINO skips unavailable plugins gracefully.
import openvino
openvino_libs_dir = os.path.join(os.path.dirname(openvino.__file__), 'libs')
for fname in os.listdir(openvino_libs_dir):
    if fname.endswith('.lib') or 'npu' in fname.lower():
        continue
    datas.append((os.path.join(openvino_libs_dir, fname), 'openvino/libs'))

binaries = []
binaries += collect_dynamic_libs('openvino_genai')
binaries += collect_dynamic_libs('openvino_tokenizers')

# 2. Add the 'styles' and 'accessible' folders to restore NVDA compatibility
styles_dir = os.path.join(plugins_dir, 'styles')
if os.path.exists(styles_dir):
    datas.append((styles_dir, 'PySide6/plugins/styles'))

accessible_dir = os.path.join(plugins_dir, 'accessible')
if os.path.exists(accessible_dir):
    datas.append((accessible_dir, 'PySide6/plugins/accessible'))

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'keyring.backends.Windows',
        'keyring.backends.macOS',
        'keyring.backends.SecretService',
        'openvino',
        'openvino_genai',
        'openvino_tokenizers'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Exclude bloated/unused packages, and moviepy/imageio/Pillow (their imageio-ffmpeg
    # dependency bundles a GPL-licensed ffmpeg.exe; the app no longer uses moviepy)
    excludes=['matplotlib', 'tkinter', 'pandas', 'scipy', 'numpy.random._examples',
              'moviepy', 'imageio', 'imageio_ffmpeg', 'PIL'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

# 1. Notice 'exclude_binaries=True'. This stops it from building a huge 1-file EXE.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Cinevoice',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# 2. Add the COLLECT command to gather everything into a fast-launching folder
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cinevoice',
)
