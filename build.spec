# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs

binaries = collect_dynamic_libs('mediapipe')

datas = [
    ('cam_head_tracker/assets/*', 'cam_head_tracker/assets'),
]

a = Analysis(
    ['cam_head_tracker/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CamHeadTracker',
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
    icon='cam_head_tracker/assets/icon.png'
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CamHeadTracker',
)


import shutil
from pathlib import Path

dist_root = Path("dist", coll.name)

if dist_root.exists():
    print("--- Copying files to root ---")
    shutil.copy2("LICENSE.txt", dist_root / "LICENSE.txt")
    shutil.copy2("NOTICE.txt", dist_root / "NOTICE.txt")
    shutil.copy2("ffmpeg-builder/LICENSE_FFMPEG.txt", dist_root / "LICENSE_FFMPEG.txt")
    print("--- Done ---")
