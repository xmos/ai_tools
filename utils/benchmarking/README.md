# Generating an xsim trace file

To run `xsim` with tracing enabled, run the following command:

>xsim --trace --enable-fnop-tracing --trace-to path/to/your/trace.out /path/to/your/app.xe

If your application requires arguments, run the following command:

> xsim --trace --enable-fnop-tracing --trace-to path/to/your/trace.out --args /path/to/your/app.xe your_arguments


# Generating a benchmark report

To print a benchmark report, run the folllowing command:

> python process_trace.py -t path/to/your/trace.out

Run `process_trace.py` with the `--help` option to discover some additional features.

# Running xsim_bench

There is a shell script, `xsim_bench`, that will run xsim and generate a report.

> ./xsim_bench.sh /path/to/your/app.xe path/to/your/trace.out

or

> ./xsim_bench.sh /path/to/your/app.xe path/to/your/trace.out "your arguments"
