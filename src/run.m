addpath("./DSST/");
addpath("./MOSSE/");

run_DSST();
mosse();

pyrun("import sys; sys.path.append('./KCF')");
pyrunfile("./KCF/run.py");