// RUN: marco %s %S/ExternalFunctionTestsLibraries/ExternalCLibrary.o --omc-bypass --model=Fact --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=2 --time-step=0.5 --precision=6 | FileCheck %s

// CHECK: "time","x"
// CHECK: 0.000000,0.000000
// CHECK: 0.500000,60.000000
// CHECK: 1.000000,120.000000
// CHECK: 1.500000,180.000000
// CHECK: 2.000000,240.000000

function externalFactorial
	input Integer x;
	output Real ris;
	external "C"
		ris = factorial(x);
end externalFactorial;
 
model Fact
	Real x(start = 0, fixed = true);
equation
	der(x) = externalFactorial(5);
end Fact;