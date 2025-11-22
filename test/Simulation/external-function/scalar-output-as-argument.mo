// RUN: cc -c %S/scalar-output-as-argument.c -o %basename_t_lib.o
// RUN: marco %s %basename_t_lib.o --omc-bypass --model=Test --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.5 --precision=6 | FileCheck %s

// CHECK: "time","x"
// CHECK: 0.000000,10.000000
// CHECK: 1.000000,10.000000

function integerOutput
	output Integer x;
external integerScalarOutputAsArgument(x);
end integerOutput;

function realOutput
	output Real x;
external realScalarOutputAsArgument(x);
end realOutput;

model Test
    Integer x;
    Real y;
equation
    x = integerOutput();
    y = realOutput();
end Test;
