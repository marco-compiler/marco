// RUN: marco --omc-bypass --model=ImplicitKepler --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1]","x[2]"
// CHECK-NEXT: 0.0000,3.6577,3.6577
// CHECK-NEXT: 0.1000,3.6577,3.6577
// CHECK-NEXT: 0.2000,3.6577,3.6577
// CHECK-NEXT: 0.3000,3.6577,3.6577
// CHECK-NEXT: 0.4000,3.6577,3.6577
// CHECK-NEXT: 0.5000,3.6577,3.6577
// CHECK-NEXT: 0.6000,3.6577,3.6577
// CHECK-NEXT: 0.7000,3.6577,3.6577
// CHECK-NEXT: 0.8000,3.6577,3.6577
// CHECK-NEXT: 0.9000,3.6577,3.6577
// CHECK-NEXT: 1.0000,3.6577,3.6577

model ImplicitKepler
	Real[2] x(each start = 3.6, fixed = false);
equation
	for i in 1:2 loop
		5.0 = x[i] - 2.72 * sin(x[i]);
	end for;
end ImplicitKepler;
