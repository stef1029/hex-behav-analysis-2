# Other labs' tools for analysing spikes against behaviour

What exists for analysing units against behaviour, what each tool does, and whether to
use it here. Maintenance figures were checked on **2026-10-12** against PyPI release
histories and GitHub activity; recheck them before adopting anything long-term.

Every tool listed works on what `build_nwb --batch` produces: spikes, trials and pose
on one clock in one NWB file. Use them in analysis code, and bring the results back as
columns or tidy tables. Plotters stay pure functions of a table and never take a
library's own objects.

**Environment note:** the current analysis env is Python 3.10 with pynwb 2.3.3. NeMoS
needs Python ≥3.12, and the latest scikit-learn needs ≥3.11. Use a separate,
version-pinned env for the tools below, and keep the existing one for the pipeline until
it has been checked under a newer pynwb.

## Verdicts at a glance

| Verdict | Tools |
|---|---|
| **Use** | pynapple, scikit-learn, SpikeInterface, zetapy, CEBRA, dynamax, Elephant, keypoint-MoSeq, statsmodels, NWB Inspector, DANDI CLI, Neurosift, BrainGlobe |
| **Use, pin the version** (good but pre-1.0) | NeMoS, Bombcell |
| **Prefer an alternative** | ssm → dynamax · tensortools → TensorLy · pymer4 → statsmodels · nwbwidgets → Neurosift · HERBS → BrainGlobe |
| **Copy the code, don't depend on it** | dPCA; write the hierarchical bootstrap in `plotting/utils/stats.py` |
| **Read, don't install** | ibllib / brainbox, AllenSDK |

## Plumbing: spikes, epochs and time series

**pynapple** (Peyrache lab): *Use, pin the version.*
Objects for spike trains (`TsGroup`), epochs (`IntervalSet`) and time series (`Tsd`),
plus the operations between them:
- restrict anything to epochs (trials, cue windows, blocks)
- bin spikes into a time × neurons matrix
- read a variable's value at each spike (head bearing at every spike)
- tuning curves normalised by occupancy
- peri-event alignment, cross-correlograms, Bayesian decoding

Reads NWB directly. 0.11.4 (Aug 2026), about 9 releases a year, widely used. It is
still pre-1.0, so function names shift between minor versions. Needs pynwb ≥2.0. It
does not know about `keep` or `neuron_id`; those are just metadata columns to filter on.

**Elephant** (NeuralEnsemble / EBRAINS): *Use for what pynapple lacks.*
Spike-train statistics: correlation measures, unitary events, GPFA and other
population methods. Mature and funded, 1.2.1 (Jun 2026). It uses its own neo/quantities
object model, so expect some conversion work.

## Single-neuron questions

**zetapy** (ZETA test, Montijn et al.): *Use.*
Tests whether a neuron responds to an event without choosing bin sizes or windows. It
is the defensible way to make a `responsive` label per neuron per event. Small project
with one main maintainer (4.1, Jun 2026), but the method is published, widely cited and
needs only numpy, scipy and matplotlib.

**NeMoS** (Flatiron): *Use, pin the version, separate env.*
GLMs for spike trains: spikes as a function of head bearing, cue, choice, reward and
spike history, with basis functions and regularisation. This is the standard way to
ask what drives a neuron. Built to pair with pynapple. Actively developed but young
(0.2.9, 129 open issues/PRs). Needs Python ≥3.12 and JAX.

**SpikeInterface template and quality metrics**: *Use.*
Waveform features (peak–trough width, half-width, repolarisation slope) give the
narrow/wide-spiking cell-type split. It also computes the quality metrics already
carried in `units.csv`. Very active (0.105.1, Oct 2026), with frequent API changes, so
pin it.

## Unit quality

**Bombcell** (Peters lab, Julie Fabre): *Try on a few sessions before relying on it.*
Classifies units by quality automatically, against published thresholds. Only
hand-curated `good` units are used here (`mua` in the 2603 curation is uncurated
noise), so its value would be a first pass over the sessions not yet curated. Very responsive maintainer, but the Python port is pre-1.0 and changing fast
(0.92, 26 releases in the last year). Check it accepts this pipeline's KS4/Phy layout.

## Population analyses

**scikit-learn**: *Use.*
Cross-validated decoders (logistic regression, SVM, LDA), PCA and permutation tests,
applied to matrices pivoted from the neuron-trial table. The standard library.

**dPCA** (Machens lab): *Copy the code into this repo.*
Splits population activity into components for cue, choice and time, which suits this
task. The method is standard, but the package has not had a PyPI release since 2020.
The code is small; copy it in rather than depend on the package, and expect to patch
older numpy idioms.

**CEBRA** (Mathis lab): *Use when it's wanted.*
Joint embeddings of neural activity and behaviour, such as pose. Comes from the
DeepLabCut lab and is designed to sit beside DLC output. Healthy (0.6.1, May 2026;
1.1k stars). Needs PyTorch.

**dynamax** (probml): *Use for latent-state models.*
HMMs and switching linear dynamical systems for hidden states in population activity.
Active (1.0.2, Jun 2026; 1k stars). JAX-based and lower-level than ssm.

**ssm** (Linderman lab): *Use dynamax instead.*
The same family of models with a friendlier interface, but dormant: one PyPI release,
in 2019, and no repository activity since May 2025.

**tensortools** (Williams): *Use TensorLy instead.*
Tensor decomposition of neurons × time × trials, which shows slow changes across trials
and sessions. Both forks have been quiet since 2023/2024. TensorLy is the maintained
general library; its activity was not checked here.

**Rastermap** (Stringer / Pachitariu): *Low priority.*
Sorts neurons so a raster of many cells shows its structure. Stable but quiet (1.0,
Oct 2024). Built for populations far larger than the 1–15 good units per session here.

## Behaviour

**keypoint-MoSeq** (Datta lab): *Use when you want behavioural states.*
Splits DLC pose into behavioural "syllables" that neural activity can be aligned to.
Active (0.6.10, Aug 2026). JAX and heavy compute, so a GPU helps.

## Statistics

The replicate question applies here as everywhere: neurons from one mouse are not
independent animals.

**Hierarchical bootstrap** (Saravanan et al. 2020): *Write it ourselves.*
Resamples mice, then sessions, then neurons. It is the accepted answer to
pseudoreplication. There is no maintained package; it is about 30 lines as a test in
`plotting/utils/stats.py`.

**statsmodels**: *Use for mixed models.*
Linear mixed models with mouse, session or neuron as random effects. Active (0.15.0,
Aug 2026). Its MixedLM is limited: crossed random effects are awkward.

**pymer4**: *Avoid.*
Wraps R's lme4, which is more capable, but it has had no release since May 2025 and
installing it through rpy2 is painful. If lme4 is needed, call R directly.

## Viewing, checking and sharing NWB files

| Tool | Verdict | What for |
|---|---|---|
| **Neurosift** | Use | Browser viewer for NWB files: units, trials and pose without writing code. Actively developed |
| **nwbwidgets** | Avoid | The older in-Jupyter viewer; dormant since 2023 |
| **NWB Inspector** | Use | Best-practice checks on files before sharing. Official, active (0.7.2, May 2026). Not installed in the current env |
| **DANDI CLI** | Use | Uploading to the DANDI archive. Very active |

## Where the probe was (histology)

**BrainGlobe / brainreg**: *Use when histology arrives.*
Registers histology to an atlas so each channel gets a brain region. A `region` column
is far more useful for splitting than raw depth, especially while the probe wiring is
unverified. Well-run organisation, active (1.0.16, Aug 2026).

**HERBS**: *Prefer BrainGlobe.* Quiet since Feb 2024, with 35 open issues.

## Reference code to read, not install

**ibllib / brainbox** (International Brain Laboratory): their single-unit and
behaviour-locked analyses for a large standardised decision task: rasters, PSTHs and
task-modulation tests. Maintained, but built around IBL's file format and
infrastructure. Borrow the methods.

**AllenSDK**: worked examples of unit and behaviour analyses on NWB. No release since
Nov 2023, 385 open issues/PRs, and it pins old dependencies. Keep it out of the
analysis env.
