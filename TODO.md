## TODO:

- [ ] determine extra run time for accessing parameters differently - can be slower?
- [ ] replace '.' with '_' when writing hyper parameters to table
- [ ] refactor get/set parameters to be more flexible - use any kernel - add a name: GPR-kernel_name ?
- [ ] try binning on 1d: use fix bin size/count
- [ ] add hyper parameter plot to inline / colab example
- [ ] add configuration (JSON) examples in documentation 
- [ ] Check can update Tensorflow / GPFlow to latest version without causing breaks
- [ ] Update setup.py to handle package installs - specifically handle different environments
- [ ] Update this README.md file, point to examples
- [ ] used argparse to read in configuration files / parameters to scripts instead of sys.argv
- [X] Allowable output types. How to save and load hyperparameters/variational parameters (individual?). Best database?
- [ ] Examples: sea ice, ocean elevation, simulated data
- [ ] Complete unit testing (pytests).
- [ ] Specify which gpytorch version should be used.
- [ ] update unit tests to remxraove sources of warnings