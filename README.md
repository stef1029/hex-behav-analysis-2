# hex-behav-analysis-2

`hba2`: from what the hex behaviour rig records to one table of trials. It puts the rig's
DAQ file, video, weight log and trial log on one clock, adds head angles from
DeepLabCut, writes one NWB file per session, and reads them back as a pandas table with
one row per trial.

Written for the lab that runs these rigs, and public so it is easy to share. The
package's own documentation starts at [hba2/README.md](hba2/README.md).

## Installing

With [uv](https://docs.astral.sh/uv/):

```
git clone https://github.com/stef1029/hex-behav-analysis-2
cd hex-behav-analysis-2
uv venv --python 3.10 .venv
uv pip install -e .                                  # loading and post-processing
uv pip install -e ".[dlc]" --torch-backend=cu128     # plus DeepLabCut, for the cluster
```

`[dlc]` pins one DeepLabCut version, so everyone's pose comes from the same one. A
tagged release (`git checkout v0.4.0`) pins the rest.

```python
from hba2 import Cohort, load_sessions, qc_trials

cohort = Cohort("/path/to/cohort", batch="2026-10_v1")
trials = qc_trials(load_sessions(cohort, include=True), min_ear_likelihood=0.6)
```

## Where this comes from

Developed in a private repository and copied here by a sync script, so each commit here
is one sync and lists the changes it carries. Pull requests can't be merged directly;
open an issue instead.
