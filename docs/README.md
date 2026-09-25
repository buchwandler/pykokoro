# Documentation guide

The source documentation is written in MyST Markdown and built with Sphinx. Install the
requirements and build HTML from this directory:

```bash
pip install -r requirements.txt
make html
```

The generated site is written to `_build/html/`. On Windows, use `python make.py html`
or `make.bat html`. Run `make clean` to remove generated files.

## Source pages

- `index.md` — engine ownership boundary and documentation index
- `quickstart.md` — plain prepared-text synthesis
- `installation.md` — runtime providers, frontends, and model assets
- `basic_usage.md` — request, configuration, and result patterns
- `advanced_features.md` — pronunciation overrides, annotations, routing, and
  calibration
- `pipeline_stages.md` — request rendering lifecycle
- `api_reference.md` — public request-centric API
- `examples.md` — maintained example scripts
- `languages.md` — language routing and model profiles
- `breaking-change-0.10.0.md` — migration and breaking-change release note
- `changelog.md` — generated historical changelog; update it through Releaseledger

Keep examples on the public `KokoroSynthesizer` / `SynthesisSegment` API. PyKokoro does
not own document parsing, SSMD interpretation, planning, or timeline composition.
