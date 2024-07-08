// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Unknown variable identifier inpt_var. Did you mean input_var?

function Test
    input Real input_var;
    output Real output_var;
algorithm
    if inpt_var == 0 then
        output_var := 1;
    end if;
end Test;
