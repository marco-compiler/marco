// RUN: marco %s ./ExternalFunctionTestsLibraries/newCLibrary.o --omc-bypass --model=TestMySum --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=1 --precision=6 | FileCheck %s

// CHECK: "time","result"
// CHECK:  0.000000, 5.000000, 15.000000

function mySum
  input Integer x[:];   
  output Integer s;
external "C" 
    s = arraySum(x);
end mySum;


model TestMySum
  Integer v[5] = {1, 2, 3, 4, 5}; 
  Integer result;
equation
  result = mySum(v);
end TestMySum;