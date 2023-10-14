// RUN: marco --omc-bypass --model=InductionUsage --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]","x[4]","x[5]"
// CHECK-NEXT: 0.0000,0.5000,0.5000,0.5000,0.5000,0.5000
// CHECK-NEXT: 0.1000,0.5200,0.5800,0.6200,0.6600,0.7000
// CHECK-NEXT: 0.2000,0.5400,0.6600,0.7400,0.8200,0.9000
// CHECK-NEXT: 0.3000,0.5600,0.7400,0.8600,0.9800,1.1000
// CHECK-NEXT: 0.4000,0.5800,0.8200,0.9800,1.1400,1.3000
// CHECK-NEXT: 0.5000,0.6000,0.9000,1.1000,1.3000,1.5000
// CHECK-NEXT: 0.6000,0.6200,0.9800,1.2200,1.4600,1.7000
// CHECK-NEXT: 0.7000,0.6400,1.0600,1.3400,1.6200,1.9000
// CHECK-NEXT: 0.8000,0.6600,1.1400,1.4600,1.7800,2.1000
// CHECK-NEXT: 0.9000,0.6800,1.2200,1.5800,1.9400,2.3000
// CHECK-NEXT: 1.0000,0.7000,1.3000,1.7000,2.1000,2.5000

model InductionUsage
	Real[5] x(each start = 0.5, fixed = true);
equation
	5.0 * der(x[1]) = 1.0;

	for i in 2:5 loop
		5.0 * der(x[i]) = 2.0 * i;
	end for;
end InductionUsage;
