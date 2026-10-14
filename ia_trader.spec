# -*- mode: python ; coding: utf-8 -*-
"""
ia_trader.spec - Especificação de Empacotamento PyInstaller para o IA Trader.
Fase 10: Gera binário standalone autocontido para Linux e Windows.
"""

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Diretório raiz
project_root = os.path.abspath(os.getcwd())

# Coleta de submódulos internos e pacotes
hidden_imports = [
    'flask',
    'flask_cors',
    'flask_jwt_extended',
    'sqlite3',
    'hashlib',
    'secrets',
    'numpy',
    'pandas',
    'ta',
    'lightgbm',
    'sklearn',
    'sklearn.ensemble',
    'sklearn.model_selection',
    'aiohttp',
    'dotenv',
    'agents',
    'agents.base_agent',
    'agents.regime_classifier',
    'agents.specialists',
    'data',
    'data.data_loader',
    'data.data_integrations',
    'data.feature_engineering',
    'data.market_structure',
    'consensus_judge',
    'signal_generator',
    'core',
    'smart_trade',
    'performance_tracker',
    'rollback_guard',
    'learning_tracker',
    'scheduler',
    'pair_scanner',
    'educational_module',
    'voice_commands',
    'auth_module',
    'flask_interface'
]

# Adiciona submódulos automáticos de pacotes críticos
hidden_imports += collect_submodules('agents')
hidden_imports += collect_submodules('data')

# Arquivos de dados (templates, licença, docs essenciais)
datas = [
    (os.path.join(project_root, 'templates'), 'templates'),
    (os.path.join(project_root, 'LICENSE.txt'), '.'),
]

# Exclusão de bibliotecas de teste e ferramentas de desenvolvimento pesadas
excludes = [
    'pytest',
    'playwright',
    'pip',
    'setuptools',
    'wheel',
    'tkinter',
    'matplotlib',
    'test',
    'tests'
]

a = Analysis(
    ['launcher.py'],
    pathex=[project_root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ia_trader',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ia_trader'
)
