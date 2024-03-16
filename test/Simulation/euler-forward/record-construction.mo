// RUN: marco --omc-bypass --model=Test --solver=euler-forward -o %basename_t -L %runtime_lib_dir %s -L %sundials_lib_dir -Wl,-rpath,%sundials_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","z1.im","z1.re","z2.im","z2.re"
// CHECK: 0.000000,2.000000,1.000000,4.000000,3.000000
// CHECK: 1.000000,2.000000,1.000000,4.000000,3.000000

model 'Test'
    function 'Complex.\'constructor\'.fromReal' "Construct Complex from Real"
        input Real 're';
        input Real 'im' = 0.0;
        output 'Complex' 'result';
    algorithm
        annotation(Inline = true);
    end 'Complex.\'constructor\'.fromReal';

    record 'Complex'
        parameter Real 're';
        parameter Real 'im';
    end 'Complex';

    parameter 'Complex' 'z1' = 'Complex'(1.0, 2.0);
    'Complex' 'z2';
equation
    'z2' = 'Complex'(3.0, 4.0);
end 'Test';
