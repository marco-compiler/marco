// RUN: marco %s %S/ExternalFunctionTestsLibraries/ExternalCLibrary.o --omc-bypass --model=Inverter --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=1 --precision=6 | FileCheck %s

// CHECK: "time","y"
// CHECK: 0.000000,0.500000
// CHECK: 1.000000,0.500000

function externalInverse
	input Real x;
	output Real ris;
	external "C"
		ris = inverse(x);
end externalInverse;
 
model Inverter
	Real y; 
equation
 	y  = externalInverse(2);
end Inverter;
