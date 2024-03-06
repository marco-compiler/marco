// RUN: marco --omc-bypass --model=TimeUsage --variable-filter="time;x;der(x)" --solver=ida -o %basename_t -L %runtime_lib_dir -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir %s
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=4 | FileCheck %s

// CHECK: "time","x"
// CHECK-NEXT: 0.0000,0.0000,0.0000
// CHECK-NEXT: 0.1000,0.0050,0.1000
// CHECK-NEXT: 0.2000,0.0200,0.2000
// CHECK-NEXT: 0.3000,0.0450,0.3000
// CHECK-NEXT: 0.4000,0.0800,0.4000
// CHECK-NEXT: 0.5000,0.1250,0.5000
// CHECK-NEXT: 0.6000,0.1800,0.6000
// CHECK-NEXT: 0.7000,0.2450,0.7000
// CHECK-NEXT: 0.8000,0.3200,0.8000
// CHECK-NEXT: 0.9000,0.4050,0.9000
// CHECK-NEXT: 1.0000,0.5000,1.0000

model TimeUsage
	Real x(start = 0, fixed = true);
equation
	der(x) = time;
end TimeUsage;
