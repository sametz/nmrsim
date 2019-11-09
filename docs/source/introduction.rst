Introduction to nmrsim v0.3.0 (beta)
=====================================

**nmrsim** is a library of tools for simulating NMR spectra, starting from
parameters provided by the user (e.g. chemical shift; *J* coupling constants;
rate constants for DNMR spectra). Currently, the application is limited to
spin-:math:`1/2` nuclei only, but expanding this to other nuclei is feasible.

The target niche for **nmrsim** are users that want to model NMR spectra but
who either are not specialists themselves, and/or who want to model NMR spectra
and concepts (e.g. spin Hamiltonians) for instructional purposes. If there is a
feature that you would like to see, or a barrier to you using this library,
feel free to open an issue on GitHub or to send the author email
(sametz at udel dot edu).

The project is inspired by the program `WINDNMR <https://www.chem.wisc.edu/areas/reich/plt/windnmr.htm>`_ by Hans
Reich. The goal for Version 1.0 of **nmrsim** is to provide Python tools for the same types of simulations that
WINDNMR did: first- and second-order simulation of spin-1/2 spin systems, plus simulation of some dynamic NMR (DNMR)
lineshapes. A longer-term goal is to expand the toolset (e.g. to allow higher-spin nuclei, or new DNMR models).

