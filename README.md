# ExhaustiveGS has been deprecated

ExhaustiveGS is an exhaustive SiDB ground state charge configuration finder, which was developed as an alternative to [siqad/simanneal-sidb](https://github.com/siqad/simanneal-sidb) for model validation.

Since then, research collaborators at the Chair for Design Automation at the Technical University of Munich have developed [QuickExact](https://www.cda.cit.tum.de/files/eda/2024_aspdac_efficient_exact_simulation.pdf) and [ClusterComplete](https://www.math.ru.nl/~bosma/Students/WillemLambooyMSc.pdf) as part of the [Munich Nanotech Toolkit](https://www.cda.cit.tum.de/research/nanotech/mnt/) (MNT). Both of them are exhaustive ground state charge configuration finders for SiDBs and significantly outperform ExhaustiveGS for all problem sizes, and are available as SiQAD plugins through [cda-tum/mnt-siqad-plugins](https://github.com/cda-tum/mnt-siqad-plugins). We thank the developers of MNT for advancing the state-of-the-art in SiDB logic simulation, and encourage users of SiQAD and ExhaustiveGS to check out their work. With this, ExhaustiveGS is deprecated and will no longer be maintained.

## Historical README (original description)
Exhaustively search for the ground state electron configuration in a given SiDB layout.

Depends on [SiQADConnector](https://github.com/retallickj/siqad/tree/master/src/phys/siqadconn) made available through SWIG from SiQAD's repository. After creating the SWIG wrapper for SiQADConnector (`siqadconn.py` and `_siqadconn.*.{so,pyd}`), copy them to the src directory.

After compiling the Python wrapper, you may use the script `wrapper_validation.py` to validate that the SWIG wrapper works as intended.
