// RUN: marco %s ../ExternalFunctionTestsLibraries/newCLibrary.o --omc-bypass --model=LogicComponent --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=4 --time-step=1| FileCheck %s

// CHECK: "time","x","y","ris"
// CHECK: 0.000000, 0, 1, 1  
// CHECK: 1.000000, 1, 1, 0
// CHECK: 2.000000, 0, 1, 1
// CHECK: 3.000000, 1, 1, 0
// CHECK: 4.000000, 0, 1, 1

function externalXorPort
	input Boolean a;
	input Boolean b;
	output Boolean ris;
	external "C"
		ris = logicXor(a,b);
end externalXor;

model LogicComponent
  Boolean x(start=false);
  Boolean y(start=true);
  Boolean ris;
equation
  when {initial(), sample(0, 1)} then
    ris = externalXorPort(x, y); 
    x   = not pre(ris);
  end when;
end LogicComponent;
