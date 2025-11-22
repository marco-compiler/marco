// RUN: marco %s --omc-bypass --model=Test --solver=euler-forward -o %basename_t %link_flags -L %runtime_lib_dir -Wl,-rpath %runtime_lib_dir
// RUN: ./%basename_t --end-time=1 --time-step=0.1 --precision=6 | FileCheck %s

// CHECK: "time","z1.im","z1.re","z2.im","z2.re","z3.im","z3.re"
// CHECK: 0.000000,2.000000,1.000000,4.000000,3.000000,6.000000,4.000000
// CHECK: 1.000000,2.000000,1.000000,4.000000,3.000000,6.000000,4.000000

model 'Test'
    function 'Complex.\'+\'' "Add two complex numbers"
        input 'Complex' 'c1';
        input 'Complex' 'c2';
        output 'Complex' 'c3';
    algorithm
        'c3' := 'Complex.\'constructor\'.fromReal'('c1'.'re' + 'c2'.'re', 'c1'.'im' + 'c2'.'im');
        annotation(Inline = true);
    end 'Complex.\'+\'';

    function 'Complex.\'constructor\'.fromReal' "Construct Complex from Real"
        input Real 're';
        input Real 'im' = 0.0;
        output 'Complex' 'result';
    algorithm
        annotation(Inline = true);
    end 'Complex.\'constructor\'.fromReal';

    record 'Complex'
        Real 're';
        Real 'im';
    end 'Complex';

    'Complex' 'z1' = 'Complex'(1.0, 2.0);
    'Complex' 'z2' = 'Complex'(3.0, 4.0);
    'Complex' 'z3';
equation
    'z3' = 'Complex.\'+\''('z1', 'z2');
end 'Test';