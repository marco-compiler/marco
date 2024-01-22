// RUN: marco --omc-bypass --model=AlgorithmInsideModel --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s -L %sundials_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y[1]","y[2]","y[3]"
// CHECK-NEXT: 0.000000,0.000000,0.000000,10.000000,10.000000
// CHECK-NEXT: 0.100000,0.100000,0.100000,10.200000,10.100000
// CHECK-NEXT: 0.200000,0.200000,0.200000,10.400000,10.200000
// CHECK-NEXT: 0.300000,0.300000,0.300000,10.600000,10.300000
// CHECK-NEXT: 0.400000,0.400000,0.400000,10.800000,10.400000
// CHECK-NEXT: 0.500000,0.500000,0.500000,11.000000,10.500000
// CHECK-NEXT: 0.600000,0.600000,0.600000,11.200000,10.600000
// CHECK-NEXT: 0.700000,0.700000,0.700000,11.400000,10.700000
// CHECK-NEXT: 0.800000,0.800000,0.800000,11.600000,10.800000
// CHECK-NEXT: 0.900000,0.900000,0.900000,11.800000,10.900000
// CHECK-NEXT: 1.000000,1.000000,1.000000,12.000000,11.000000

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
