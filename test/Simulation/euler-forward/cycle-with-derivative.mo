// RUN: marco --omc-bypass --model=CycleWithDerivative --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir
// RUN: ./%basename_t --end-time=0.2 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK-NEXT: 0.000000,0.000000,1.000000
// CHECK-NEXT: 0.100000,0.300000,1.000000
// CHECK-NEXT: 0.200000,0.600000,1.000000

model CycleWithDerivative
	Real x(start = 0, fixed = true);
	Real y;
equation
	der(x) + y = 4;
	der(x) - y = 2;
end CycleWithDerivative;
