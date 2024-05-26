// RUN: marco %s --omc-bypass --model=AlgorithmInsideModel --solver=euler-forward -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y[1]","y[2]","y[3]"
// CHECK: 0.000000,0.000000,0.000000,10.000000,10.000000
// CHECK: 1.000000,1.000000,1.000000,12.000000,11.000000

model AlgorithmInsideModel
    Real x;
    Real[3] y(each start = 5);
equation
    x = time;
algorithm
    y[1] := x;
    y[2] := y[2] * 2;
    y[3] := y[2] + y[1];
    y[2] := y[2] + x * 2;
end AlgorithmInsideModel;
