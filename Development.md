## tsar

Per creare l'ambiente:

```
virtualenv ambiente
source ./ambiente/bin/activate
pip install -e .
```

Per provare le componenti del programma

```
source ./ambiente/bin/activate
python -m unittest
```

Per lavorare al codice,

```
source ./ambiente/bin/activate
Atom .
```

Per caricare su pip, dopo aver cambiato la versione in setup.py

```
source ./ambiente/bin/activate
bash pip_upload.sh
```

Per profilare i test
```
pip install nose-cprof snakeviz
nosetests --with-cprofile
snakeviz stats.dat
```
