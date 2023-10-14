// RUN: marco --omc-bypass --model=ArraysWithState --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","y[1]","y[2]","y[3]"
// CHECK-NEXT: 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
// CHECK-NEXT: 0.1000,0.1000,0.1000,0.1000,0.3050,0.3050,0.3050
// CHECK-NEXT: 0.2000,0.2000,0.2000,0.2000,0.6200,0.6200,0.6200
// CHECK-NEXT: 0.3000,0.3000,0.3000,0.3000,0.9450,0.9450,0.9450
// CHECK-NEXT: 0.4000,0.4000,0.4000,0.4000,1.2800,1.2800,1.2800
// CHECK-NEXT: 0.5000,0.5000,0.5000,0.5000,1.6250,1.6250,1.6250
// CHECK-NEXT: 0.6000,0.6000,0.6000,0.6000,1.9800,1.9800,1.9800
// CHECK-NEXT: 0.7000,0.7000,0.7000,0.7000,2.3450,2.3450,2.3450
// CHECK-NEXT: 0.8000,0.8000,0.8000,0.8000,2.7200,2.7200,2.7200
// CHECK-NEXT: 0.9000,0.9000,0.9000,0.9000,3.1050,3.1050,3.1050
// CHECK-NEXT: 1.0000,1.0000,1.0000,1.0000,3.5000,3.5000,3.5000

model ArraysWithState
	Real[3] x(each start = 0, fixed = true);
	Real[3] y(each start = 0, fixed = true);
equation
	for i in 1:3 loop
		der(x[i]) = 1.0;
		der(y[i]) = 3 + x[i];
	end for;
end ArraysWithState;
