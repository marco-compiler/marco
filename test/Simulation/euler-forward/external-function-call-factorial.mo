// RUN: marco %s %S/ExternalFunctionTestsLibraries/newCLibrary.o --omc-bypass --model=Fact --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=1 --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK:  0.000000, 5.000000, 120.000000

function externalFactorial
	input Integer x;
	output Integer ris;
	external "C"
		ris = factorial(x);
end externalFactorial;

model Fact
	Integer x(start = 5, fixed = true);
	Integer y;
equation
	y = externalFactorial(x);
end Fact;
