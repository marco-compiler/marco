// RUN: marco %s --omc-bypass --model=TimeUsage --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x","y"
// CHECK: 0.000000,0.000000,0.000000
// CHECK: 1.000000,-0.506366,0.023282

model TimeUsage
	Real x;
	Real y(start = 0, fixed = true);
equation
    x = sin(time * 100);
    der(y) = x;
end TimeUsage;
