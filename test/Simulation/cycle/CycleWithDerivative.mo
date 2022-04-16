// RUN: marco --omc-bypass --model=M1 --end-time=0.2 --time-step=0.1 -o %basename_t %s
// RUN: ./%basename_t | FileCheck %s

// CHECK: time;x;y
// CHECK-NEXT: 0.000000000000;0.000000000000;1.000000000000
// CHECK-NEXT: 0.100000000000;0.300000000000;1.000000000000
// CHECK-NEXT: 0.200000000000;0.600000000000;1.000000000000

model CycleWithDerivative
	Real x;
	Real y;
equation
	der(x) + y = 4;
	der(x) - y = 2;
end CycleWithDerivative;
