// RUN: marco --omc-bypass --model=M1 --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y","z"
// CHECK: 0.000000,-0.500000,3.500000,-1.500000
// CHECK: 0.200000,-0.500000,3.500000,-1.500000

model M1
    Real x;
    Real y;
    Real z;
equation
    x = y - 4;
    y = 2 - z;
    z = 3 * x;
end M1;
