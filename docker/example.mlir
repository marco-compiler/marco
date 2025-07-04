module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.stack_alignment" = 128 : i64, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, f128 = dense<128> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, !bmodelica.real = ["size", 64], !bmodelica.int = ["size", 64], i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  bmodelica.function @externalLog {
    bmodelica.variable @b : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @n : !bmodelica.variable<!bmodelica.int, input>
    bmodelica.variable @ris : !bmodelica.variable<!bmodelica.int, output>
    %0 = bmodelica.variable_get @b : !bmodelica.int
    %1 = bmodelica.variable_get @n : !bmodelica.int
    %2 = bmodelica.call @externalLog(%0, %1) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
  }
  bmodelica.model @SimpleFirstOrder  {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.real>
    bmodelica.start @x {
      %0 = bmodelica.constant #bmodelica<real 0.000000e+00> : !bmodelica.real
      bmodelica.yield %0 : !bmodelica.real
    } {each = false, fixed = true}
    bmodelica.dynamic {
      bmodelica.equation {
        %0 = bmodelica.variable_get @x : !bmodelica.real
        %1 = bmodelica.der %0 : !bmodelica.real -> !bmodelica.real
        %2 = bmodelica.constant #bmodelica<int 4> : !bmodelica.int
        %3 = bmodelica.constant #bmodelica<int 2> : !bmodelica.int
        %4 = bmodelica.constant #bmodelica<int 256> : !bmodelica.int
        %5 = bmodelica.call @externalLog(%3, %4) : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %6 = bmodelica.sub %2, %5 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
        %7 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.int>
        bmodelica.equation_sides %7, %8 : tuple<!bmodelica.real>, tuple<!bmodelica.int>
      }
    }
  }
}
