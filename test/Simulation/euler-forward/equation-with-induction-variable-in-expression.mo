// RUN: marco --omc-bypass --model=InductionInExpression --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s -L %sundials_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x[1]","x[2]","x[3]"
// CHECK-NEXT: 0.000000,1.000000,2.000000,3.000000
// CHECK-NEXT: 0.100000,1.100000,2.100000,3.100000
// CHECK-NEXT: 0.200000,1.200000,2.200000,3.200000
// CHECK-NEXT: 0.300000,1.300000,2.300000,3.300000
// CHECK-NEXT: 0.400000,1.400000,2.400000,3.400000
// CHECK-NEXT: 0.500000,1.500000,2.500000,3.500000
// CHECK-NEXT: 0.600000,1.600000,2.600000,3.600000
// CHECK-NEXT: 0.700000,1.700000,2.700000,3.700000
// CHECK-NEXT: 0.800000,1.800000,2.800000,3.800000
// CHECK-NEXT: 0.900000,1.900000,2.900000,3.900000
// CHECK-NEXT: 1.000000,2.000000,3.000000,4.000000

model InductionInExpression
	Real[3] x;
equation
	for i in 1:3 loop
		x[i] = i + time;
	end for;
end InductionInExpression;
