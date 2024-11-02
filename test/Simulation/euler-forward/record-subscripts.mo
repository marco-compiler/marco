// RUN: marco %s --omc-bypass --model=Test --solver=euler-forward -o %basename_t -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","a.x[1,1]","a.x[1,2]","a.x[2,1]","a.x[2,2]","a.x[3,1]","a.x[3,2]","y1[1]","y1[2]","y2[1]","y2[2]","y2[3]"
// CHECK: 0.000000,1.000000,2.000000,1.000000,2.000000,1.000000,2.000000,1.000000,2.000000,2.000000,2.000000,2.000000
// CHECK: 1.000000,1.000000,2.000000,1.000000,2.000000,1.000000,2.000000,1.000000,2.000000,2.000000,2.000000,2.000000

model Test
    record R
        Real[2] x;
    end R;

    R[3] a;
    Real[2] y1;
    Real[3] y2;

equation
    for i in 1:3 loop
      a[i].x = {1.0, 2.0};
    end for;

    y1 = a[2].x;
    y2 = a.x[2];
end Test;
