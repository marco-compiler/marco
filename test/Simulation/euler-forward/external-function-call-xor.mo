// RUN: marco %s %S/ExternalFunctionTestsLibraries/ExternalCLibrary.o --omc-bypass --model=LogicComponent --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=1 --precision=6| FileCheck %s

// CHECK: "time","ris_1","ris_2","ris_3","ris_4"
// CHECK: 0.000000,0.000000,1.000000,1.000000,0.000000
// CHECK: 1.000000,0.000000,1.000000,1.000000,0.000000

function externalXorPort
  input Boolean a;
  input Boolean b;
  output Boolean ris;
  external "C"
    ris = logicXor(a,b);
end externalXorPort;
 
model LogicComponent
  Boolean ris_1;
  Boolean ris_2; 
  Boolean ris_3; 
  Boolean ris_4; 
equation
  ris_1 = externalXorPort(0, 0); 
  ris_2 = externalXorPort(0, 1); 
  ris_3 = externalXorPort(1, 0); 
  ris_4 = externalXorPort(1, 1); 
end LogicComponent;