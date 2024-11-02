// RUN: marco %s --omc-bypass --model=Test --solver=rk-euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","x"
// CHECK: 0.000000,0.000000
// CHECK: 1.000000,0.651322

model Test
	Real x(start = 0, fixed = true);
equation
    der(x) = 1 - x;
end Test;
