# PEM Coupling Calculator
www.pem.ligo.org
Started by Julia Kruk
Completed and maintained by Philippe Nguyen

This program performs an analysis of environmental coupling for a set of PEM sensors during different PEM injections.
It uses GWpy and PEM_coupling modules to:
    1) preprocess data from PEM sensors and DARM
    2) compute coupling functions for each sensor during each injection
    3) estimates DARM ambients for every coupling function
    4) aggregate data across multiple injections to produce a composite coupling function and estimated ambient for each sensor
    5) export data in the form of:
        a) CSV files containing single-injection coupling functions (in physical units and in counts) and estimated ambients
        b) plots of single-injection coupling functions (in physical units and in counts)
        c) spectrum plots showing sensor and DARM ASDs during and before injections, super-imposed with a estimated ambients
        d) CSV files containing composite coupling functions (in physical units and in counts) and estimated ambients
        e) plots of composite coupling functions (in physical units and in counts) and estimated ambients
        f) multi-plots of composite coupling functions and estimated ambients, labeling data by injection

GWPy:
    This code is written to run on gwpy.
    On LIGO cluster, activate the gwpy virtualenv via ". ~detchar/opt/gwpysoft/bin/activate"