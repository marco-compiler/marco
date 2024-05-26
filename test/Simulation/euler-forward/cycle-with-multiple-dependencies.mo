// RUN: marco %s --omc-bypass --model=M1 --solver=euler-forward -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir -L %llvm_lib_dir -Wl,-rpath,%llvm_lib_dir
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y","z"
// CHECK: 0.000000,-3.000000,-17.000000,2.000000
// CHECK: 0.200000,-3.000000,-17.000000,2.000000

model M1
    Real x;
    Real y;
    Real z;
equation
    x = y + z + 12;
    y = 5 * x - 2;
    z = -2 * x - 4;
end M1;
