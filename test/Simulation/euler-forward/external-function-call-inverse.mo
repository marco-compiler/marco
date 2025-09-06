// RUN: marco %s ExternalFunctionTestsLibraries/newCLibrary.o --omc-bypass --model=Inverter --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1.5 --time-step=0.5 --precision=6 | FileCheck %s

// CHECK: "time","y"
// CHECK:  0.000000, 2.000000
// CHECK:  0.500000, 3.000000
// CHECK:  1.000000, 4.500000
// CHECK:  1.500000, 6.750000

function externalInverse
	input Real x;
	output Real ris;
	external "C"
		ris = inverse(x);
end externalInverse;

model Inverter
  discrete Real y(start=1);
equation
  when {initial(), sample(0,0.5)} then
    y = y + externalInverse(y);
  end when;
end Inverter;


