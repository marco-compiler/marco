module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e", "dlti.endianness" = "little", !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !bmodelica.real = ["size", 64], !bmodelica.int = ["size", 64]>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  bmodelica.external_function @discreteLog {sym_visibility = "private"} : (!bmodelica.int, !bmodelica.int) -> !bmodelica.real
  bmodelica.function @externalLogReal {
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @ris : !bmodelica.variable<!bmodelica.real, output>
    bmodelica.algorithm {
      %0 = bmodelica.variable_get @b : !bmodelica.int
      %1 = bmodelica.variable_get @n : !bmodelica.int
      %2 = bmodelica.variable_get @ris : !bmodelica.real
      %3 = bmodelica.call @discreteLog(%0, %1, %2) : (!bmodelica.int, !bmodelica.int, !bmodelica.real) -> !bmodelica.real
    }
  }
  bmodelica.model @SimpleFirstOrder  {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.dynamic {
      bmodelica.equation {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
        %2 = bmodelica.constant #bmodelica<int 100> : !bmodelica.int
        %3 = bmodelica.call @externalLogReal(%1, %2) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.real
        %4 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
      }
    }
  }
}
