# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['jangjorim_client.py'],
             pathex=['/Users/jackyoung96/jackyoung_folder/2020-1/EngProduct/posenet-pytorch'],
             binaries=[("/Users/jackyoung96/opt/anaconda3/envs/poseestimate/lib/python3.7/site-packages/torch/lib/libtorch_global_deps.dylib","."),
                        ("/Users/jackyoung96/opt/anaconda3/envs/poseestimate/lib/python3.7/site-packages/PySide2/QtNetwork.abi3.so","."),
                        ("/Users/jackyoung96/opt/anaconda3/envs/poseestimate/lib/libmkl_tbb_thread.dylib", "."),
                        ("/Users/jackyoung96/opt/anaconda3/envs/poseestimate/lib/python3.7/site-packages/PySide2/QtGui.abi3.so", "."),
                        ("/Users/jackyoung96/opt/anaconda3/envs/poseestimate/lib/python3.7/site-packages/PySide2/QtCore.abi3.so", "."),
                        ("/Users/jackyoung96/opt/anaconda3/envs/poseestimate/lib/python3.7/site-packages/PySide2/libpyside2.abi3.5.15.dylib", ".")],
             datas=[("/Users/jackyoung96/jackyoung_folder/2020-1/EngProduct/posenet-pytorch/jangjorim_games/image" , 'jangjorim_games/image'),
                    ("/Users/jackyoung96/jackyoung_folder/2020-1/EngProduct/posenet-pytorch/jangjorim_games/font" , 'jangjorim_games/font'),
                    ("/Users/jackyoung96/jackyoung_folder/2020-1/EngProduct/posenet-pytorch/jangjorim_games/sound" , 'jangjorim_games/sound')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='jangjorim_client',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=True,
          icon="jumple.icns")
