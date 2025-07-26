module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", f64 = dense<64> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !bmodelica.real = ["size", 64], !bmodelica.int = ["size", 64]>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  bmodelica.function @factorial {
    bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @res : !bmodelica.variable<!bmodelica.int, output>
    bmodelica.algorithm {
      %0 = bmodelica.variable_get @res : !bmodelica.int
      %1 = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
      %2 = bmodelica.mul %0, %1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
      bmodelica.variable_set @res, %2 : !bmodelica.int
    }
  }
  bmodelica.model @FactorialGrowth  {
    bmodelica.variable @growthFactor : !bmodelica.variable<!bmodelica.int, parameter>
    bmodelica.variable @y : !bmodelica.variable<!bmodelica.real>
    bmodelica.binding_equation @growthFactor {
      %0 = bmodelica.constant #bmodelica<int 4> : !bmodelica.int
      bmodelica.yield %0 : !bmodelica.int
    }
    bmodelica.start @y {
      %0 = bmodelica.constant #bmodelica<int 0> : !bmodelica.int
      bmodelica.yield %0 : !bmodelica.int
    } {each = false, fixed = true}
    bmodelica.dynamic {
      bmodelica.equation {
        %0 = bmodelica.variable_get @y : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.variable_get @growthFactor : !bmodelica.int
        %3 = bmodelica.call @factorial(%2) : (!bmodelica.int) -> !bmodelica.int
        %4 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.int>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.int>
      }
    }
  }
}
