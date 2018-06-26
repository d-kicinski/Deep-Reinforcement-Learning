# Instalacja środowiska

```
virutalenv env
source env/bin/activate
pip install -r requirements.txt
```

W razie problemów z biblioteką PyTorch można ją zainstalować bezpośredio:

Linux:
```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-linux_x86_64.whl
pip3 install torchvision

```

Windows:
```
pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.0-cp36-cp36m-win_amd64.whl
pip3 install torchvision
```


# Uruchomienie przykładoweo skrypty z uczeniem
```
./run_traingin.sh
```
