// RUN: marco %s %S/ExternalFunctionTestsLibraries/ExternalCLibrary.o --omc-bypass --model=TestMySum --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=1 --precision=6 | FileCheck %s

// CHECK: "time","result","x[1]","x[2]","x[3]","x[4]","x[5]"
// CHECK: 0.000000,15.000000,1.000000,2.000000,3.000000,4.000000,5.000000
// CHECK: 1.000000,15.000000,1.000000,2.000000,3.000000,4.000000,5.000000

function mySum
    input Integer[:] x;
    output Integer s;
external "C" 
    s = arraySum(x);
end mySum;

model TestMySum
    Integer[5] x;
    Integer result;
equation
    x[1] = 1;
    x[2] = 2;
    x[3] = 3;
    x[4] = 4;
    x[5] = 5;
    result = mySum(x);
end TestMySum;