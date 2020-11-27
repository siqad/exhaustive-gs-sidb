# exhaustive-gs-sidb
Exhaustively search for the ground state electron configuration in a given Si-DB layout.

Depends on [SiQADConnector](https://github.com/retallickj/siqad/tree/master/src/phys/siqadconn) made available through SWIG from SiQAD's repository. After creating the SWIG wrapper for SiQADConnector (`siqadconn.py` and `_siqadconn.*.{so,pyd}`), copy them to the src directory.

After compiling the Python wrapper, you may use the script `wrapper_validation.py` to validate that the SWIG wrapper works as intended.
