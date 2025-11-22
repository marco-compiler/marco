// RUN: cc -c %S/array-argument.c -o %basename_t_lib.o
// RUN: marco %s %basename_t_lib.o --omc-bypass --model=Test --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.5 --precision=6 | FileCheck %s

// CHECK: "time","x[1,1]","x[1,2]","x[1,3]","x[1,4]","x[1,5]","x[2,1]","x[2,2]","x[2,3]","x[2,4]","x[2,5]","xSum","y[1,1]","y[1,2]","y[1,3]","y[1,4]","y[1,5]","y[2,1]","y[2,2]","y[2,3]","y[2,4]","y[2,5]","ySum"
// CHECK: 0.000000,1.000000,2.000000,3.000000,4.000000,5.000000,6.000000,7.000000,8.000000,9.000000,10.000000,55.000000,1.500000,2.500000,3.500000,4.500000,5.500000,6.500000,7.500000,8.500000,9.500000,10.500000,60.000000
// CHECK: 1.000000,1.000000,2.000000,3.000000,4.000000,5.000000,6.000000,7.000000,8.000000,9.000000,10.000000,55.000000,1.500000,2.500000,3.500000,4.500000,5.500000,6.500000,7.500000,8.500000,9.500000,10.500000,60.000000

function integerSum
	input Integer[2,5] array;
	output Integer sum;
external sum = sumIntegerArray(array);
end integerSum;

function realSum
	input Real[2,5] array;
	output Real sum;
external sum = sumRealArray(array);
end realSum;

model Test
    Integer[2,5] x;
    Real[2,5] y;
    Integer xSum;
    Real ySum;
equation
    x[1,1] = 1;
    x[1,2] = 2;
    x[1,3] = 3;
    x[1,4] = 4;
    x[1,5] = 5;
    x[2,1] = 6;
    x[2,2] = 7;
    x[2,3] = 8;
    x[2,4] = 9;
    x[2,5] = 10;

    y[1,1] = 1.5;
    y[1,2] = 2.5;
    y[1,3] = 3.5;
    y[1,4] = 4.5;
    y[1,5] = 5.5;
    y[2,1] = 6.5;
    y[2,2] = 7.5;
    y[2,3] = 8.5;
    y[2,4] = 9.5;
    y[2,5] = 10.5;

    xSum = integerSum(x);
    ySum = realSum(y);
end Test;
