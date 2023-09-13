// RUN: marco --omc-bypass --model=MultidimensionalArray --solver=ida -o %basename_t %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x[1,1]","x[1,2]","x[1,3]","x[2,1]","x[2,2]","x[2,3]"
// CHECK-NEXT: 0.0000,0.0000,0.0000,0.0000,0.0000,0.0000,0.0000
// CHECK-NEXT: 0.1000,0.1000,0.2000,0.2000,0.1000,0.2000,0.2000
// CHECK-NEXT: 0.2000,0.2000,0.4000,0.4000,0.2000,0.4000,0.4000
// CHECK-NEXT: 0.3000,0.3000,0.6000,0.6000,0.3000,0.6000,0.6000
// CHECK-NEXT: 0.4000,0.4000,0.8000,0.8000,0.4000,0.8000,0.8000
// CHECK-NEXT: 0.5000,0.5000,1.0000,1.0000,0.5000,1.0000,1.0000
// CHECK-NEXT: 0.6000,0.6000,1.2000,1.2000,0.6000,1.2000,1.2000
// CHECK-NEXT: 0.7000,0.7000,1.4000,1.4000,0.7000,1.4000,1.4000
// CHECK-NEXT: 0.8000,0.8000,1.6000,1.6000,0.8000,1.6000,1.6000
// CHECK-NEXT: 0.9000,0.9000,1.8000,1.8000,0.9000,1.8000,1.8000
// CHECK-NEXT: 1.0000,1.0000,2.0000,2.0000,1.0000,2.0000,2.0000

model MultidimensionalArray
	Real[2, 3] x(each start = 0, fixed = true);
equation
	for i in 1:2 loop
		der(x[i, 1]) = 1.0;

		for j in 2:3 loop
			der(x[i, j]) = 2.0;
		end for;
	end for;
end MultidimensionalArray;
