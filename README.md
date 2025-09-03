# SEISMIC SECTION MATCHING

Spectral, STFT and CWT seismic section matching.

## 🚀 Get and install

```bash
git clone https://github.com/sergeevsn/seismatch_py.git
cd seismatch_py
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 📦 Requirements

- ```numpy```

- ```scipy```

- ```matplotlib```

- ```ssqueezepy``` for CWT

## 🧪 Usage

Change file names with yours in scripts:

```python
source_segy_file = 'input.sgy'
reference_segy_file = 'reference.sgy'
output_segy_file = 'matched_cwt.sgy'
```

and then launch preferred script.

The best result achieved when reference's time and trace count match source's
