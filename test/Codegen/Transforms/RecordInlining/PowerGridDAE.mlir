module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<"dlti.endianness", "little">, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  modelica.model @PowerGridDAE.GridBase {
    modelica.function @"ComplexPU.'+'" attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c3 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.add %1, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %6 = modelica.component_get %5, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %7 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %8 = modelica.component_get %7, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %9 = modelica.add %6, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %10 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%4, %9) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c3, %10 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @"PowerGridDAE.Complex.'+'" attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c3 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.add %1, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %6 = modelica.component_get %5, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %7 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %8 = modelica.component_get %7, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %9 = modelica.add %6, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %10 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%4, %9) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c3, %10 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @"PowerGridDAE.Complex.'*'.multiply" attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c3 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.mul %1, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %6 = modelica.component_get %5, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %7 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %8 = modelica.component_get %7, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %9 = modelica.mul %6, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %10 = modelica.sub %4, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %11 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %12 = modelica.component_get %11, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %13 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %14 = modelica.component_get %13, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %15 = modelica.mul %12, %14 : (!modelica.real, !modelica.real) -> !modelica.real
        %16 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %17 = modelica.component_get %16, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %18 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %19 = modelica.component_get %18, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %20 = modelica.mul %17, %19 : (!modelica.real, !modelica.real) -> !modelica.real
        %21 = modelica.add %15, %20 : (!modelica.real, !modelica.real) -> !modelica.real
        %22 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%10, %21) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c3, %22 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @"PowerGridDAE.Complex.'-'.subtract" attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c3 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.sub %1, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %6 = modelica.component_get %5, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %7 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %8 = modelica.component_get %7, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %9 = modelica.sub %6, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %10 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%4, %9) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c3, %10 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @"PowerGridDAE.Complex.'constructor'.fromReal" attributes {inline = true} {
      modelica.variable @re : !modelica.variable<!modelica.real, input>
      modelica.variable @im : !modelica.variable<!modelica.real, input>
      modelica.variable @result : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.default @im {
        %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        modelica.yield %0 : !modelica.real
      }
      modelica.algorithm {
        %0 = modelica.variable_get @re : !modelica.variable<!modelica.real, input>
        %1 = modelica.variable_get @im : !modelica.variable<!modelica.real, input>
        %2 = modelica.record_create %0, %1 : !modelica.variable<!modelica.real, input>, !modelica.variable<!modelica.real, input> -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @result, %2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @PowerGridDAE.ComplexMath.conj attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.neg %3 : !modelica.real -> !modelica.real
        %5 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%1, %4) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c2, %5 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @PowerGridDAE.ComplexMath.fromPolar attributes {inline = true} {
      modelica.variable @len : !modelica.variable<!modelica.real, input>
      modelica.variable @phi : !modelica.variable<!modelica.real, input>
      modelica.variable @c : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @len : !modelica.real
        %1 = modelica.variable_get @phi : !modelica.real
        %2 = modelica.cos %1 : !modelica.real -> !modelica.real
        %3 = modelica.mul %0, %2 : (!modelica.real, !modelica.real) -> !modelica.real
        %4 = modelica.variable_get @len : !modelica.real
        %5 = modelica.variable_get @phi : !modelica.real
        %6 = modelica.sin %5 : !modelica.real -> !modelica.real
        %7 = modelica.mul %4, %6 : (!modelica.real, !modelica.real) -> !modelica.real
        %8 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%3, %7) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c, %8 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @PowerGridDAE.ComplexMath.real attributes {inline = true} {
      modelica.variable @c : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @r : !modelica.variable<!modelica.real, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        modelica.variable_set @r, %1 : !modelica.real
      }
    }
    modelica.function @"PowerGridDAE.GridBase.ComplexPU.'*'.multiply" attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c3 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.mul %1, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %6 = modelica.component_get %5, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %7 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %8 = modelica.component_get %7, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %9 = modelica.mul %6, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %10 = modelica.sub %4, %9 : (!modelica.real, !modelica.real) -> !modelica.real
        %11 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %12 = modelica.component_get %11, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %13 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %14 = modelica.component_get %13, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %15 = modelica.mul %12, %14 : (!modelica.real, !modelica.real) -> !modelica.real
        %16 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %17 = modelica.component_get %16, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %18 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %19 = modelica.component_get %18, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %20 = modelica.mul %17, %19 : (!modelica.real, !modelica.real) -> !modelica.real
        %21 = modelica.add %15, %20 : (!modelica.real, !modelica.real) -> !modelica.real
        %22 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%10, %21) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c3, %22 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @"PowerGridDAE.GridBase.ComplexPU.'-'.subtract" attributes {inline = true} {
      modelica.variable @c1 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c2 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, input>
      modelica.variable @c3 : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.algorithm {
        %0 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %1 = modelica.component_get %0, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %2 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %3 = modelica.component_get %2, @re : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %4 = modelica.sub %1, %3 : (!modelica.real, !modelica.real) -> !modelica.real
        %5 = modelica.variable_get @c1 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %6 = modelica.component_get %5, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %7 = modelica.variable_get @c2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %8 = modelica.component_get %7, @im : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex> -> !modelica.real
        %9 = modelica.sub %6, %8 : (!modelica.real, !modelica.real) -> !modelica.real
        %10 = modelica.call @"PowerGridDAE.Complex.'constructor'.fromReal"(%4, %9) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @c3, %10 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.function @"PowerGridDAE.GridBase.ComplexPU.'constructor'.fromReal" attributes {inline = true} {
      modelica.variable @re : !modelica.variable<!modelica.real, input>
      modelica.variable @im : !modelica.variable<!modelica.real, input>
      modelica.variable @result : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, output>
      modelica.default @im {
        %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        modelica.yield %0 : !modelica.real
      }
      modelica.algorithm {
        %0 = modelica.variable_get @re : !modelica.variable<!modelica.real, input>
        %1 = modelica.variable_get @im : !modelica.variable<!modelica.real, input>
        %2 = modelica.record_create %0, %1 : !modelica.variable<!modelica.real, input>, !modelica.variable<!modelica.real, input> -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        modelica.variable_set @result, %2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      }
    }
    modelica.record @ComplexPU {
      modelica.variable @re : !modelica.variable<!modelica.real, constant>
      modelica.variable @im : !modelica.variable<!modelica.real, constant>
    }
    modelica.record @PowerGridDAE.Complex {
      modelica.variable @re : !modelica.variable<!modelica.real, output>
      modelica.variable @im : !modelica.variable<!modelica.real, output>
    }
    modelica.variable @j : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, constant>
    modelica.variable @Ne : !modelica.variable<!modelica.int, parameter>
    modelica.variable @P : !modelica.variable<!modelica.real, parameter>
    modelica.variable @R : !modelica.variable<!modelica.real, parameter>
    modelica.variable @V : !modelica.variable<!modelica.real, parameter>
    modelica.variable @X : !modelica.variable<!modelica.real, parameter>
    modelica.variable @Ta : !modelica.variable<!modelica.real, parameter>
    modelica.variable @sigma : !modelica.variable<!modelica.real, parameter>
    modelica.variable @f_n : !modelica.variable<!modelica.real, parameter>
    modelica.variable @pi : !modelica.variable<!modelica.real, constant>
    modelica.variable @N : !modelica.variable<!modelica.int, parameter>
    modelica.variable @Ng : !modelica.variable<!modelica.int, parameter>
    modelica.variable @Y : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, parameter>
    modelica.variable @Vg : !modelica.variable<!modelica.real, parameter>
    modelica.variable @Vl : !modelica.variable<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, parameter>
    modelica.variable @omega_n : !modelica.variable<!modelica.real, parameter>
    modelica.variable @i_n : !modelica.variable<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    modelica.variable @v : !modelica.variable<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    modelica.variable @i_h : !modelica.variable<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    modelica.variable @i_v : !modelica.variable<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    modelica.variable @theta : !modelica.variable<4x2x!modelica.real>
    modelica.variable @omega : !modelica.variable<4x2x!modelica.real>
    modelica.variable @Pg : !modelica.variable<4x2x!modelica.real>
    modelica.variable @Pm : !modelica.variable<4x2x!modelica.real>
    modelica.variable @v_out : !modelica.variable<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, output>
    modelica.variable @omega_out : !modelica.variable<4x!modelica.real, output>
    modelica.variable @i_n_start : !modelica.variable<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, parameter>
    modelica.variable @v_start : !modelica.variable<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, parameter>
    modelica.binding_equation @j {
      %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %1 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
      %2 = modelica.record_create %0, %1 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      modelica.yield %2 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
    }
    modelica.binding_equation @Ne {
      %0 = modelica.constant #modelica.int<2> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.binding_equation @P {
      %0 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @R {
      %0 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @V {
      %0 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @X {
      %0 = modelica.constant #modelica.real<3.000000e-01> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @Ta {
      %0 = modelica.constant #modelica.real<5.000000e+00> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @sigma {
      %0 = modelica.constant #modelica.real<5.000000e-02> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @f_n {
      %0 = modelica.constant #modelica.real<5.000000e+01> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @pi {
      %0 = modelica.constant #modelica.real<3.1415926535897931> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @N {
      %0 = modelica.constant #modelica.int<4> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.binding_equation @Ng {
      %0 = modelica.constant #modelica.int<2> : !modelica.int
      modelica.yield %0 : !modelica.int
    }
    modelica.binding_equation @Y {
      %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %1 = modelica.constant #modelica.real<3.333333333333333> : !modelica.real
      %2 = modelica.neg %1 : !modelica.real -> !modelica.real
      %3 = modelica.record_create %0, %2 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      modelica.yield %3 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
    }
    modelica.binding_equation @Vg {
      %0 = modelica.constant #modelica.real<1.0028085560065789> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.binding_equation @Vl {
      %0 = modelica.constant #modelica.real<1.005271584208677> : !modelica.real
      %1 = modelica.constant #modelica.real<0.01884884220391269> : !modelica.real
      %2 = modelica.neg %1 : !modelica.real -> !modelica.real
      %3 = modelica.record_create %0, %2 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      modelica.yield %3 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
    }
    modelica.binding_equation @omega_n {
      %0 = modelica.constant #modelica.real<314.15926535897933> : !modelica.real
      modelica.yield %0 : !modelica.real
    }
    modelica.start @theta {
      %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      modelica.yield %0 : !modelica.real
    } {each = false, fixed = true}
    modelica.start @omega {
      %0 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
      modelica.yield %0 : !modelica.real
    } {each = false, fixed = true}
    modelica.equation {
      %0 = modelica.constant #modelica.int<1> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.constant #modelica.int<1> : !modelica.int
      %4 = modelica.constant -1 : index
      %5 = modelica.add %3, %4 : (!modelica.int, index) -> index
      %6 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %7 = modelica.subscription %6[%2, %5] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %8 = modelica.load %7[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %9 = modelica.constant #modelica.int<1> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.constant #modelica.int<1> : !modelica.int
      %13 = modelica.constant -1 : index
      %14 = modelica.add %12, %13 : (!modelica.int, index) -> index
      %15 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.subscription %15[%11, %14] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %17 = modelica.load %16[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %18 = modelica.call @"ComplexPU.'+'"(%8, %17) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %19 = modelica.constant #modelica.int<1> : !modelica.int
      %20 = modelica.constant -1 : index
      %21 = modelica.add %19, %20 : (!modelica.int, index) -> index
      %22 = modelica.constant #modelica.int<1> : !modelica.int
      %23 = modelica.constant -1 : index
      %24 = modelica.add %22, %23 : (!modelica.int, index) -> index
      %25 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %26 = modelica.subscription %25[%21, %24] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %27 = modelica.load %26[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %28 = modelica.call @"PowerGridDAE.Complex.'+'"(%18, %27) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %29 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %30 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %31 = modelica.record_create %29, %30 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %32 = modelica.equation_side %28 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      %33 = modelica.equation_side %31 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      modelica.equation_sides %32, %33 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<4> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.constant #modelica.int<1> : !modelica.int
      %4 = modelica.constant -1 : index
      %5 = modelica.add %3, %4 : (!modelica.int, index) -> index
      %6 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %7 = modelica.subscription %6[%2, %5] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %8 = modelica.load %7[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %9 = modelica.constant #modelica.int<3> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.constant #modelica.int<1> : !modelica.int
      %13 = modelica.constant -1 : index
      %14 = modelica.add %12, %13 : (!modelica.int, index) -> index
      %15 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.subscription %15[%11, %14] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %17 = modelica.load %16[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %18 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'-'.subtract"(%8, %17) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %19 = modelica.constant #modelica.int<4> : !modelica.int
      %20 = modelica.constant -1 : index
      %21 = modelica.add %19, %20 : (!modelica.int, index) -> index
      %22 = modelica.constant #modelica.int<1> : !modelica.int
      %23 = modelica.constant -1 : index
      %24 = modelica.add %22, %23 : (!modelica.int, index) -> index
      %25 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %26 = modelica.subscription %25[%21, %24] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %27 = modelica.load %26[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %28 = modelica.call @"PowerGridDAE.Complex.'+'"(%18, %27) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %29 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %30 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %31 = modelica.record_create %29, %30 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %32 = modelica.equation_side %28 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      %33 = modelica.equation_side %31 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      modelica.equation_sides %32, %33 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<1> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.constant #modelica.int<4> : !modelica.int
      %4 = modelica.constant -1 : index
      %5 = modelica.add %3, %4 : (!modelica.int, index) -> index
      %6 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %7 = modelica.subscription %6[%2, %5] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %8 = modelica.load %7[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %9 = modelica.constant #modelica.int<1> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.constant #modelica.int<4> : !modelica.int
      %13 = modelica.constant -1 : index
      %14 = modelica.add %12, %13 : (!modelica.int, index) -> index
      %15 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.subscription %15[%11, %14] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %17 = modelica.load %16[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %18 = modelica.call @"ComplexPU.'+'"(%8, %17) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %19 = modelica.constant #modelica.int<1> : !modelica.int
      %20 = modelica.constant -1 : index
      %21 = modelica.add %19, %20 : (!modelica.int, index) -> index
      %22 = modelica.constant #modelica.int<3> : !modelica.int
      %23 = modelica.constant -1 : index
      %24 = modelica.add %22, %23 : (!modelica.int, index) -> index
      %25 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %26 = modelica.subscription %25[%21, %24] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %27 = modelica.load %26[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %28 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%18, %27) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %29 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %30 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %31 = modelica.record_create %29, %30 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %32 = modelica.equation_side %28 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      %33 = modelica.equation_side %31 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      modelica.equation_sides %32, %33 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<4> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.constant #modelica.int<4> : !modelica.int
      %4 = modelica.constant -1 : index
      %5 = modelica.add %3, %4 : (!modelica.int, index) -> index
      %6 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %7 = modelica.subscription %6[%2, %5] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %8 = modelica.load %7[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %9 = modelica.constant #modelica.int<3> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.constant #modelica.int<4> : !modelica.int
      %13 = modelica.constant -1 : index
      %14 = modelica.add %12, %13 : (!modelica.int, index) -> index
      %15 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.subscription %15[%11, %14] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %17 = modelica.load %16[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %18 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'-'.subtract"(%8, %17) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %19 = modelica.constant #modelica.int<4> : !modelica.int
      %20 = modelica.constant -1 : index
      %21 = modelica.add %19, %20 : (!modelica.int, index) -> index
      %22 = modelica.constant #modelica.int<3> : !modelica.int
      %23 = modelica.constant -1 : index
      %24 = modelica.add %22, %23 : (!modelica.int, index) -> index
      %25 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %26 = modelica.subscription %25[%21, %24] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %27 = modelica.load %26[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %28 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%18, %27) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %29 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %30 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
      %31 = modelica.record_create %29, %30 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
      %32 = modelica.equation_side %28 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      %33 = modelica.equation_side %31 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      modelica.equation_sides %32, %33 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<1> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @v_out : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %5 = modelica.load %4[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %6 = modelica.constant #modelica.int<1> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<1> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %14 = modelica.load %13[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %15 = modelica.equation_side %5 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.equation_side %14 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      modelica.equation_sides %15, %16 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<2> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @v_out : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %5 = modelica.load %4[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %6 = modelica.constant #modelica.int<1> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<4> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %14 = modelica.load %13[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %15 = modelica.equation_side %5 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.equation_side %14 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      modelica.equation_sides %15, %16 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<3> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @v_out : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %5 = modelica.load %4[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %6 = modelica.constant #modelica.int<4> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<1> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %14 = modelica.load %13[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %15 = modelica.equation_side %5 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.equation_side %14 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      modelica.equation_sides %15, %16 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<4> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @v_out : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %5 = modelica.load %4[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %6 = modelica.constant #modelica.int<4> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<4> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %14 = modelica.load %13[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %15 = modelica.equation_side %5 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      %16 = modelica.equation_side %14 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
      modelica.equation_sides %15, %16 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<1> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @omega_out : !modelica.array<4x!modelica.real>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.real>
      %5 = modelica.load %4[] : !modelica.array<!modelica.real>
      %6 = modelica.constant #modelica.int<1> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<1> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x2x!modelica.real>
      %14 = modelica.load %13[] : !modelica.array<!modelica.real>
      %15 = modelica.equation_side %5 : tuple<!modelica.real>
      %16 = modelica.equation_side %14 : tuple<!modelica.real>
      modelica.equation_sides %15, %16 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<2> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @omega_out : !modelica.array<4x!modelica.real>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.real>
      %5 = modelica.load %4[] : !modelica.array<!modelica.real>
      %6 = modelica.constant #modelica.int<1> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<2> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x2x!modelica.real>
      %14 = modelica.load %13[] : !modelica.array<!modelica.real>
      %15 = modelica.equation_side %5 : tuple<!modelica.real>
      %16 = modelica.equation_side %14 : tuple<!modelica.real>
      modelica.equation_sides %15, %16 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<3> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @omega_out : !modelica.array<4x!modelica.real>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.real>
      %5 = modelica.load %4[] : !modelica.array<!modelica.real>
      %6 = modelica.constant #modelica.int<4> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<1> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x2x!modelica.real>
      %14 = modelica.load %13[] : !modelica.array<!modelica.real>
      %15 = modelica.equation_side %5 : tuple<!modelica.real>
      %16 = modelica.equation_side %14 : tuple<!modelica.real>
      modelica.equation_sides %15, %16 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.equation {
      %0 = modelica.constant #modelica.int<4> : !modelica.int
      %1 = modelica.constant -1 : index
      %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
      %3 = modelica.variable_get @omega_out : !modelica.array<4x!modelica.real>
      %4 = modelica.subscription %3[%2] : !modelica.array<4x!modelica.real>
      %5 = modelica.load %4[] : !modelica.array<!modelica.real>
      %6 = modelica.constant #modelica.int<4> : !modelica.int
      %7 = modelica.constant -1 : index
      %8 = modelica.add %6, %7 : (!modelica.int, index) -> index
      %9 = modelica.constant #modelica.int<2> : !modelica.int
      %10 = modelica.constant -1 : index
      %11 = modelica.add %9, %10 : (!modelica.int, index) -> index
      %12 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
      %13 = modelica.subscription %12[%8, %11] : !modelica.array<4x2x!modelica.real>
      %14 = modelica.load %13[] : !modelica.array<!modelica.real>
      %15 = modelica.equation_side %5 : tuple<!modelica.real>
      %16 = modelica.equation_side %14 : tuple<!modelica.real>
      modelica.equation_sides %15, %16 : tuple<!modelica.real>, tuple<!modelica.real>
    }
    modelica.for_equation %arg0 = 1 to 4 {
      modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation {
          %0 = modelica.constant #modelica.real<5.000000e+00> : !modelica.real
          %1 = modelica.constant -1 : index
          %2 = modelica.add %arg0, %1 : (index, index) -> index
          %3 = modelica.constant -1 : index
          %4 = modelica.add %arg1, %3 : (index, index) -> index
          %5 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
          %6 = modelica.subscription %5[%2, %4] : !modelica.array<4x2x!modelica.real>
          %7 = modelica.load %6[] : !modelica.array<!modelica.real>
          %8 = modelica.mul %0, %7 : (!modelica.real, !modelica.real) -> !modelica.real
          %9 = modelica.constant -1 : index
          %10 = modelica.add %arg0, %9 : (index, index) -> index
          %11 = modelica.constant -1 : index
          %12 = modelica.add %arg1, %11 : (index, index) -> index
          %13 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
          %14 = modelica.subscription %13[%10, %12] : !modelica.array<4x2x!modelica.real>
          %15 = modelica.load %14[] : !modelica.array<!modelica.real>
          %16 = modelica.der %15 : !modelica.real -> !modelica.real
          %17 = modelica.mul %8, %16 : (!modelica.real, !modelica.real) -> !modelica.real
          %18 = modelica.constant -1 : index
          %19 = modelica.add %arg0, %18 : (index, index) -> index
          %20 = modelica.constant -1 : index
          %21 = modelica.add %arg1, %20 : (index, index) -> index
          %22 = modelica.variable_get @Pm : !modelica.array<4x2x!modelica.real>
          %23 = modelica.subscription %22[%19, %21] : !modelica.array<4x2x!modelica.real>
          %24 = modelica.load %23[] : !modelica.array<!modelica.real>
          %25 = modelica.constant -1 : index
          %26 = modelica.add %arg0, %25 : (index, index) -> index
          %27 = modelica.constant -1 : index
          %28 = modelica.add %arg1, %27 : (index, index) -> index
          %29 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
          %30 = modelica.subscription %29[%26, %28] : !modelica.array<4x2x!modelica.real>
          %31 = modelica.load %30[] : !modelica.array<!modelica.real>
          %32 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
          %33 = modelica.sub %31, %32 : (!modelica.real, !modelica.real) -> !modelica.real
          %34 = modelica.constant #modelica.real<5.000000e-02> : !modelica.real
          %35 = modelica.div %33, %34 : (!modelica.real, !modelica.real) -> !modelica.real
          %36 = modelica.sub %24, %35 : (!modelica.real, !modelica.real) -> !modelica.real
          %37 = modelica.constant -1 : index
          %38 = modelica.add %arg0, %37 : (index, index) -> index
          %39 = modelica.constant -1 : index
          %40 = modelica.add %arg1, %39 : (index, index) -> index
          %41 = modelica.variable_get @Pg : !modelica.array<4x2x!modelica.real>
          %42 = modelica.subscription %41[%38, %40] : !modelica.array<4x2x!modelica.real>
          %43 = modelica.load %42[] : !modelica.array<!modelica.real>
          %44 = modelica.sub %36, %43 : (!modelica.real, !modelica.real) -> !modelica.real
          %45 = modelica.equation_side %17 : tuple<!modelica.real>
          %46 = modelica.equation_side %44 : tuple<!modelica.real>
          modelica.equation_sides %45, %46 : tuple<!modelica.real>, tuple<!modelica.real>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 4 {
      modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant -1 : index
          %3 = modelica.add %arg1, %2 : (index, index) -> index
          %4 = modelica.variable_get @theta : !modelica.array<4x2x!modelica.real>
          %5 = modelica.subscription %4[%1, %3] : !modelica.array<4x2x!modelica.real>
          %6 = modelica.load %5[] : !modelica.array<!modelica.real>
          %7 = modelica.der %6 : !modelica.real -> !modelica.real
          %8 = modelica.constant -1 : index
          %9 = modelica.add %arg0, %8 : (index, index) -> index
          %10 = modelica.constant -1 : index
          %11 = modelica.add %arg1, %10 : (index, index) -> index
          %12 = modelica.variable_get @omega : !modelica.array<4x2x!modelica.real>
          %13 = modelica.subscription %12[%9, %11] : !modelica.array<4x2x!modelica.real>
          %14 = modelica.load %13[] : !modelica.array<!modelica.real>
          %15 = modelica.constant #modelica.real<1.000000e+00> : !modelica.real
          %16 = modelica.sub %14, %15 : (!modelica.real, !modelica.real) -> !modelica.real
          %17 = modelica.constant #modelica.real<314.15926535897933> : !modelica.real
          %18 = modelica.mul %16, %17 : (!modelica.real, !modelica.real) -> !modelica.real
          %19 = modelica.equation_side %7 : tuple<!modelica.real>
          %20 = modelica.equation_side %18 : tuple<!modelica.real>
          modelica.equation_sides %19, %20 : tuple<!modelica.real>, tuple<!modelica.real>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 4 step 2 {
      modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant #modelica.int<2> : !modelica.int
          %3 = modelica.mul %2, %arg1 : (!modelica.int, index) -> !modelica.int
          %4 = modelica.constant #modelica.int<1> : !modelica.int
          %5 = modelica.sub %3, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.constant -1 : index
          %7 = modelica.add %5, %6 : (!modelica.int, index) -> index
          %8 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %9 = modelica.subscription %8[%1, %7] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %10 = modelica.load %9[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %11 = modelica.constant -1 : index
          %12 = modelica.add %arg0, %11 : (index, index) -> index
          %13 = modelica.constant #modelica.int<2> : !modelica.int
          %14 = modelica.mul %13, %arg1 : (!modelica.int, index) -> !modelica.int
          %15 = modelica.constant #modelica.int<1> : !modelica.int
          %16 = modelica.sub %14, %15 : (!modelica.int, !modelica.int) -> !modelica.int
          %17 = modelica.constant -1 : index
          %18 = modelica.add %16, %17 : (!modelica.int, index) -> index
          %19 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %20 = modelica.subscription %19[%12, %18] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %21 = modelica.load %20[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %22 = modelica.call @PowerGridDAE.ComplexMath.conj(%21) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %23 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'*'.multiply"(%10, %22) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %24 = modelica.call @PowerGridDAE.ComplexMath.real(%23) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.real
          %25 = modelica.constant -1 : index
          %26 = modelica.add %arg0, %25 : (index, index) -> index
          %27 = modelica.constant -1 : index
          %28 = modelica.add %arg1, %27 : (index, index) -> index
          %29 = modelica.variable_get @Pg : !modelica.array<4x2x!modelica.real>
          %30 = modelica.subscription %29[%26, %28] : !modelica.array<4x2x!modelica.real>
          %31 = modelica.load %30[] : !modelica.array<!modelica.real>
          %32 = modelica.neg %31 : !modelica.real -> !modelica.real
          %33 = modelica.equation_side %24 : tuple<!modelica.real>
          %34 = modelica.equation_side %32 : tuple<!modelica.real>
          modelica.equation_sides %33, %34 : tuple<!modelica.real>, tuple<!modelica.real>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 4 step 2 {
      modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant #modelica.int<2> : !modelica.int
          %3 = modelica.mul %2, %arg1 : (!modelica.int, index) -> !modelica.int
          %4 = modelica.constant #modelica.int<1> : !modelica.int
          %5 = modelica.sub %3, %4 : (!modelica.int, !modelica.int) -> !modelica.int
          %6 = modelica.constant -1 : index
          %7 = modelica.add %5, %6 : (!modelica.int, index) -> index
          %8 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %9 = modelica.subscription %8[%1, %7] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %10 = modelica.load %9[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %11 = modelica.constant #modelica.real<1.0028085560065789> : !modelica.real
          %12 = modelica.constant -1 : index
          %13 = modelica.add %arg0, %12 : (index, index) -> index
          %14 = modelica.constant -1 : index
          %15 = modelica.add %arg1, %14 : (index, index) -> index
          %16 = modelica.variable_get @theta : !modelica.array<4x2x!modelica.real>
          %17 = modelica.subscription %16[%13, %15] : !modelica.array<4x2x!modelica.real>
          %18 = modelica.load %17[] : !modelica.array<!modelica.real>
          %19 = modelica.call @PowerGridDAE.ComplexMath.fromPolar(%11, %18) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %20 = modelica.equation_side %10 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %21 = modelica.equation_side %19 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %20, %21 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
    modelica.for_equation %arg0 = 2 to 4 step 2 {
      modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant #modelica.int<2> : !modelica.int
          %3 = modelica.mul %2, %arg1 : (!modelica.int, index) -> !modelica.int
          %4 = modelica.constant -1 : index
          %5 = modelica.add %3, %4 : (!modelica.int, index) -> index
          %6 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.subscription %6[%1, %5] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %8 = modelica.load %7[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %9 = modelica.constant -1 : index
          %10 = modelica.add %arg0, %9 : (index, index) -> index
          %11 = modelica.constant #modelica.int<2> : !modelica.int
          %12 = modelica.mul %11, %arg1 : (!modelica.int, index) -> !modelica.int
          %13 = modelica.constant -1 : index
          %14 = modelica.add %12, %13 : (!modelica.int, index) -> index
          %15 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %16 = modelica.subscription %15[%10, %14] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %17 = modelica.load %16[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %18 = modelica.call @PowerGridDAE.ComplexMath.conj(%17) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %19 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'*'.multiply"(%8, %18) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %20 = modelica.call @PowerGridDAE.ComplexMath.real(%19) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.real
          %21 = modelica.constant -1 : index
          %22 = modelica.add %arg0, %21 : (index, index) -> index
          %23 = modelica.constant -1 : index
          %24 = modelica.add %arg1, %23 : (index, index) -> index
          %25 = modelica.variable_get @Pg : !modelica.array<4x2x!modelica.real>
          %26 = modelica.subscription %25[%22, %24] : !modelica.array<4x2x!modelica.real>
          %27 = modelica.load %26[] : !modelica.array<!modelica.real>
          %28 = modelica.neg %27 : !modelica.real -> !modelica.real
          %29 = modelica.equation_side %20 : tuple<!modelica.real>
          %30 = modelica.equation_side %28 : tuple<!modelica.real>
          modelica.equation_sides %29, %30 : tuple<!modelica.real>, tuple<!modelica.real>
        }
      }
    }
    modelica.for_equation %arg0 = 2 to 4 step 2 {
      modelica.for_equation %arg1 = 1 to 2 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant #modelica.int<2> : !modelica.int
          %3 = modelica.mul %2, %arg1 : (!modelica.int, index) -> !modelica.int
          %4 = modelica.constant -1 : index
          %5 = modelica.add %3, %4 : (!modelica.int, index) -> index
          %6 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.subscription %6[%1, %5] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %8 = modelica.load %7[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %9 = modelica.constant #modelica.real<1.0028085560065789> : !modelica.real
          %10 = modelica.constant -1 : index
          %11 = modelica.add %arg0, %10 : (index, index) -> index
          %12 = modelica.constant -1 : index
          %13 = modelica.add %arg1, %12 : (index, index) -> index
          %14 = modelica.variable_get @theta : !modelica.array<4x2x!modelica.real>
          %15 = modelica.subscription %14[%11, %13] : !modelica.array<4x2x!modelica.real>
          %16 = modelica.load %15[] : !modelica.array<!modelica.real>
          %17 = modelica.call @PowerGridDAE.ComplexMath.fromPolar(%9, %16) : (!modelica.real, !modelica.real) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %18 = modelica.equation_side %8 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %19 = modelica.equation_side %17 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %18, %19 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 3 {
      modelica.for_equation %arg1 = 1 to 4 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant -1 : index
          %3 = modelica.add %arg1, %2 : (index, index) -> index
          %4 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %5 = modelica.subscription %4[%1, %3] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %6 = modelica.load %5[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.constant -1 : index
          %8 = modelica.add %arg0, %7 : (index, index) -> index
          %9 = modelica.constant -1 : index
          %10 = modelica.add %arg1, %9 : (index, index) -> index
          %11 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %12 = modelica.subscription %11[%8, %10] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %13 = modelica.load %12[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %14 = modelica.constant #modelica.int<1> : !modelica.int
          %15 = modelica.add %arg0, %14 : (index, !modelica.int) -> !modelica.int
          %16 = modelica.constant -1 : index
          %17 = modelica.add %15, %16 : (!modelica.int, index) -> index
          %18 = modelica.constant -1 : index
          %19 = modelica.add %arg1, %18 : (index, index) -> index
          %20 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %21 = modelica.subscription %20[%17, %19] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %22 = modelica.load %21[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %23 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'-'.subtract"(%13, %22) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %24 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
          %25 = modelica.constant #modelica.real<3.333333333333333> : !modelica.real
          %26 = modelica.neg %25 : !modelica.real -> !modelica.real
          %27 = modelica.record_create %24, %26 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %28 = modelica.call @"PowerGridDAE.Complex.'*'.multiply"(%23, %27) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %29 = modelica.equation_side %6 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %30 = modelica.equation_side %28 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %29, %30 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 4 {
      modelica.for_equation %arg1 = 1 to 3 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant -1 : index
          %3 = modelica.add %arg1, %2 : (index, index) -> index
          %4 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %5 = modelica.subscription %4[%1, %3] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %6 = modelica.load %5[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.constant -1 : index
          %8 = modelica.add %arg0, %7 : (index, index) -> index
          %9 = modelica.constant -1 : index
          %10 = modelica.add %arg1, %9 : (index, index) -> index
          %11 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %12 = modelica.subscription %11[%8, %10] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %13 = modelica.load %12[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %14 = modelica.constant -1 : index
          %15 = modelica.add %arg0, %14 : (index, index) -> index
          %16 = modelica.constant #modelica.int<1> : !modelica.int
          %17 = modelica.add %arg1, %16 : (index, !modelica.int) -> !modelica.int
          %18 = modelica.constant -1 : index
          %19 = modelica.add %17, %18 : (!modelica.int, index) -> index
          %20 = modelica.variable_get @v : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %21 = modelica.subscription %20[%15, %19] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %22 = modelica.load %21[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %23 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'-'.subtract"(%13, %22) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %24 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
          %25 = modelica.constant #modelica.real<3.333333333333333> : !modelica.real
          %26 = modelica.neg %25 : !modelica.real -> !modelica.real
          %27 = modelica.record_create %24, %26 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %28 = modelica.call @"PowerGridDAE.Complex.'*'.multiply"(%23, %27) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %29 = modelica.equation_side %6 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %30 = modelica.equation_side %28 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %29, %30 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
    modelica.for_equation %arg0 = 2 to 3 {
      modelica.equation {
        %0 = modelica.constant -1 : index
        %1 = modelica.add %arg0, %0 : (index, index) -> index
        %2 = modelica.constant #modelica.int<1> : !modelica.int
        %3 = modelica.constant -1 : index
        %4 = modelica.add %2, %3 : (!modelica.int, index) -> index
        %5 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %6 = modelica.subscription %5[%1, %4] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %7 = modelica.load %6[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %8 = modelica.constant -1 : index
        %9 = modelica.add %arg0, %8 : (index, index) -> index
        %10 = modelica.constant #modelica.int<1> : !modelica.int
        %11 = modelica.constant -1 : index
        %12 = modelica.add %10, %11 : (!modelica.int, index) -> index
        %13 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %14 = modelica.subscription %13[%9, %12] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %15 = modelica.load %14[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %16 = modelica.call @"ComplexPU.'+'"(%7, %15) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %17 = modelica.constant -1 : index
        %18 = modelica.add %arg0, %17 : (index, index) -> index
        %19 = modelica.constant #modelica.int<1> : !modelica.int
        %20 = modelica.constant -1 : index
        %21 = modelica.add %19, %20 : (!modelica.int, index) -> index
        %22 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %23 = modelica.subscription %22[%18, %21] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %24 = modelica.load %23[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %25 = modelica.call @"PowerGridDAE.Complex.'+'"(%16, %24) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %26 = modelica.constant #modelica.int<1> : !modelica.int
        %27 = modelica.sub %arg0, %26 : (index, !modelica.int) -> !modelica.int
        %28 = modelica.constant -1 : index
        %29 = modelica.add %27, %28 : (!modelica.int, index) -> index
        %30 = modelica.constant #modelica.int<1> : !modelica.int
        %31 = modelica.constant -1 : index
        %32 = modelica.add %30, %31 : (!modelica.int, index) -> index
        %33 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %34 = modelica.subscription %33[%29, %32] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %35 = modelica.load %34[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %36 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%25, %35) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %37 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %38 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %39 = modelica.record_create %37, %38 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %40 = modelica.equation_side %36 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        %41 = modelica.equation_side %39 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        modelica.equation_sides %40, %41 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      }
    }
    modelica.for_equation %arg0 = 2 to 3 {
      modelica.equation {
        %0 = modelica.constant -1 : index
        %1 = modelica.add %arg0, %0 : (index, index) -> index
        %2 = modelica.constant #modelica.int<4> : !modelica.int
        %3 = modelica.constant -1 : index
        %4 = modelica.add %2, %3 : (!modelica.int, index) -> index
        %5 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %6 = modelica.subscription %5[%1, %4] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %7 = modelica.load %6[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %8 = modelica.constant -1 : index
        %9 = modelica.add %arg0, %8 : (index, index) -> index
        %10 = modelica.constant #modelica.int<4> : !modelica.int
        %11 = modelica.constant -1 : index
        %12 = modelica.add %10, %11 : (!modelica.int, index) -> index
        %13 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %14 = modelica.subscription %13[%9, %12] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %15 = modelica.load %14[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %16 = modelica.call @"ComplexPU.'+'"(%7, %15) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %17 = modelica.constant -1 : index
        %18 = modelica.add %arg0, %17 : (index, index) -> index
        %19 = modelica.constant #modelica.int<3> : !modelica.int
        %20 = modelica.constant -1 : index
        %21 = modelica.add %19, %20 : (!modelica.int, index) -> index
        %22 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %23 = modelica.subscription %22[%18, %21] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %24 = modelica.load %23[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %25 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%16, %24) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %26 = modelica.constant #modelica.int<1> : !modelica.int
        %27 = modelica.sub %arg0, %26 : (index, !modelica.int) -> !modelica.int
        %28 = modelica.constant -1 : index
        %29 = modelica.add %27, %28 : (!modelica.int, index) -> index
        %30 = modelica.constant #modelica.int<4> : !modelica.int
        %31 = modelica.constant -1 : index
        %32 = modelica.add %30, %31 : (!modelica.int, index) -> index
        %33 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %34 = modelica.subscription %33[%29, %32] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %35 = modelica.load %34[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %36 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%25, %35) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %37 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %38 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %39 = modelica.record_create %37, %38 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %40 = modelica.equation_side %36 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        %41 = modelica.equation_side %39 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        modelica.equation_sides %40, %41 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      }
    }
    modelica.for_equation %arg0 = 2 to 3 {
      modelica.equation {
        %0 = modelica.constant #modelica.int<1> : !modelica.int
        %1 = modelica.constant -1 : index
        %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
        %3 = modelica.constant -1 : index
        %4 = modelica.add %arg0, %3 : (index, index) -> index
        %5 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %6 = modelica.subscription %5[%2, %4] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %7 = modelica.load %6[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %8 = modelica.constant #modelica.int<1> : !modelica.int
        %9 = modelica.constant -1 : index
        %10 = modelica.add %8, %9 : (!modelica.int, index) -> index
        %11 = modelica.constant -1 : index
        %12 = modelica.add %arg0, %11 : (index, index) -> index
        %13 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %14 = modelica.subscription %13[%10, %12] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %15 = modelica.load %14[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %16 = modelica.call @"ComplexPU.'+'"(%7, %15) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %17 = modelica.constant #modelica.int<1> : !modelica.int
        %18 = modelica.constant -1 : index
        %19 = modelica.add %17, %18 : (!modelica.int, index) -> index
        %20 = modelica.constant -1 : index
        %21 = modelica.add %arg0, %20 : (index, index) -> index
        %22 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %23 = modelica.subscription %22[%19, %21] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %24 = modelica.load %23[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %25 = modelica.call @"PowerGridDAE.Complex.'+'"(%16, %24) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %26 = modelica.constant #modelica.int<1> : !modelica.int
        %27 = modelica.constant -1 : index
        %28 = modelica.add %26, %27 : (!modelica.int, index) -> index
        %29 = modelica.constant #modelica.int<1> : !modelica.int
        %30 = modelica.sub %arg0, %29 : (index, !modelica.int) -> !modelica.int
        %31 = modelica.constant -1 : index
        %32 = modelica.add %30, %31 : (!modelica.int, index) -> index
        %33 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %34 = modelica.subscription %33[%28, %32] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %35 = modelica.load %34[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %36 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%25, %35) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %37 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %38 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %39 = modelica.record_create %37, %38 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %40 = modelica.equation_side %36 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        %41 = modelica.equation_side %39 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        modelica.equation_sides %40, %41 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      }
    }
    modelica.for_equation %arg0 = 2 to 3 {
      modelica.equation {
        %0 = modelica.constant #modelica.int<4> : !modelica.int
        %1 = modelica.constant -1 : index
        %2 = modelica.add %0, %1 : (!modelica.int, index) -> index
        %3 = modelica.constant -1 : index
        %4 = modelica.add %arg0, %3 : (index, index) -> index
        %5 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %6 = modelica.subscription %5[%2, %4] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %7 = modelica.load %6[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %8 = modelica.constant #modelica.int<3> : !modelica.int
        %9 = modelica.constant -1 : index
        %10 = modelica.add %8, %9 : (!modelica.int, index) -> index
        %11 = modelica.constant -1 : index
        %12 = modelica.add %arg0, %11 : (index, index) -> index
        %13 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %14 = modelica.subscription %13[%10, %12] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %15 = modelica.load %14[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %16 = modelica.call @"PowerGridDAE.GridBase.ComplexPU.'-'.subtract"(%7, %15) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %17 = modelica.constant #modelica.int<4> : !modelica.int
        %18 = modelica.constant -1 : index
        %19 = modelica.add %17, %18 : (!modelica.int, index) -> index
        %20 = modelica.constant -1 : index
        %21 = modelica.add %arg0, %20 : (index, index) -> index
        %22 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %23 = modelica.subscription %22[%19, %21] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %24 = modelica.load %23[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %25 = modelica.call @"PowerGridDAE.Complex.'+'"(%16, %24) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %26 = modelica.constant #modelica.int<4> : !modelica.int
        %27 = modelica.constant -1 : index
        %28 = modelica.add %26, %27 : (!modelica.int, index) -> index
        %29 = modelica.constant #modelica.int<1> : !modelica.int
        %30 = modelica.sub %arg0, %29 : (index, !modelica.int) -> !modelica.int
        %31 = modelica.constant -1 : index
        %32 = modelica.add %30, %31 : (!modelica.int, index) -> index
        %33 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %34 = modelica.subscription %33[%28, %32] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %35 = modelica.load %34[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
        %36 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%25, %35) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %37 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %38 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
        %39 = modelica.record_create %37, %38 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
        %40 = modelica.equation_side %36 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        %41 = modelica.equation_side %39 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        modelica.equation_sides %40, %41 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
      }
    }
    modelica.for_equation %arg0 = 2 to 3 {
      modelica.for_equation %arg1 = 2 to 3 {
        modelica.equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant -1 : index
          %3 = modelica.add %arg1, %2 : (index, index) -> index
          %4 = modelica.variable_get @i_n : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %5 = modelica.subscription %4[%1, %3] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %6 = modelica.load %5[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.constant -1 : index
          %8 = modelica.add %arg0, %7 : (index, index) -> index
          %9 = modelica.constant -1 : index
          %10 = modelica.add %arg1, %9 : (index, index) -> index
          %11 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %12 = modelica.subscription %11[%8, %10] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %13 = modelica.load %12[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %14 = modelica.call @"ComplexPU.'+'"(%6, %13) : (!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %15 = modelica.constant -1 : index
          %16 = modelica.add %arg0, %15 : (index, index) -> index
          %17 = modelica.constant -1 : index
          %18 = modelica.add %arg1, %17 : (index, index) -> index
          %19 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %20 = modelica.subscription %19[%16, %18] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %21 = modelica.load %20[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %22 = modelica.call @"PowerGridDAE.Complex.'+'"(%14, %21) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %23 = modelica.constant #modelica.int<1> : !modelica.int
          %24 = modelica.sub %arg0, %23 : (index, !modelica.int) -> !modelica.int
          %25 = modelica.constant -1 : index
          %26 = modelica.add %24, %25 : (!modelica.int, index) -> index
          %27 = modelica.constant -1 : index
          %28 = modelica.add %arg1, %27 : (index, index) -> index
          %29 = modelica.variable_get @i_h : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %30 = modelica.subscription %29[%26, %28] : !modelica.array<3x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %31 = modelica.load %30[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %32 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%22, %31) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %33 = modelica.constant -1 : index
          %34 = modelica.add %arg0, %33 : (index, index) -> index
          %35 = modelica.constant #modelica.int<1> : !modelica.int
          %36 = modelica.sub %arg1, %35 : (index, !modelica.int) -> !modelica.int
          %37 = modelica.constant -1 : index
          %38 = modelica.add %36, %37 : (!modelica.int, index) -> index
          %39 = modelica.variable_get @i_v : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %40 = modelica.subscription %39[%34, %38] : !modelica.array<4x3x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %41 = modelica.load %40[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %42 = modelica.call @"PowerGridDAE.Complex.'-'.subtract"(%32, %41) : (!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>, !modelica.record<@PowerGridDAE.GridBase::@ComplexPU>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %43 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
          %44 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
          %45 = modelica.record_create %43, %44 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %46 = modelica.equation_side %42 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          %47 = modelica.equation_side %45 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %46, %47 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 4 {
      modelica.for_equation %arg1 = 1 to 4 {
        modelica.initial_equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant -1 : index
          %3 = modelica.add %arg1, %2 : (index, index) -> index
          %4 = modelica.variable_get @i_n_start : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %5 = modelica.subscription %4[%1, %3] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %6 = modelica.load %5[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.add %arg0, %arg1 : (index, index) -> index
          %8 = modelica.constant #modelica.int<2> : !modelica.int
          %9 = modelica.mod %7, %8 : (index, !modelica.int) -> !modelica.int
          %10 = modelica.constant #modelica.int<0> : !modelica.int
          %11 = modelica.eq %9, %10 : (!modelica.int, !modelica.int) -> !modelica.bool
          %12 = modelica.constant #modelica.real<0.9971993098884564> : !modelica.real
          %13 = modelica.neg %12 : !modelica.real -> !modelica.real
          %14 = modelica.constant #modelica.real<0.074789948241634235> : !modelica.real
          %15 = modelica.record_create %13, %14 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %16 = modelica.constant #modelica.real<1.005271584208677> : !modelica.real
          %17 = modelica.constant #modelica.real<0.01884884220391269> : !modelica.real
          %18 = modelica.neg %17 : !modelica.real -> !modelica.real
          %19 = modelica.record_create %16, %18 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %20 = modelica.select (%11 : !modelica.bool), (%15 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>), (%19 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %21 = modelica.equation_side %6 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %22 = modelica.equation_side %20 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %21, %22 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
    modelica.for_equation %arg0 = 1 to 4 {
      modelica.for_equation %arg1 = 1 to 4 {
        modelica.initial_equation {
          %0 = modelica.constant -1 : index
          %1 = modelica.add %arg0, %0 : (index, index) -> index
          %2 = modelica.constant -1 : index
          %3 = modelica.add %arg1, %2 : (index, index) -> index
          %4 = modelica.variable_get @v_start : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %5 = modelica.subscription %4[%1, %3] : !modelica.array<4x4x!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %6 = modelica.load %5[] : !modelica.array<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %7 = modelica.add %arg0, %arg1 : (index, index) -> index
          %8 = modelica.constant #modelica.int<2> : !modelica.int
          %9 = modelica.mod %7, %8 : (index, !modelica.int) -> !modelica.int
          %10 = modelica.constant #modelica.int<0> : !modelica.int
          %11 = modelica.eq %9, %10 : (!modelica.int, !modelica.int) -> !modelica.bool
          %12 = modelica.constant #modelica.real<1.0028085560065789> : !modelica.real
          %13 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
          %14 = modelica.record_create %12, %13 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %15 = modelica.constant #modelica.real<1.005271584208677> : !modelica.real
          %16 = modelica.constant #modelica.real<0.01884884220391269> : !modelica.real
          %17 = modelica.neg %16 : !modelica.real -> !modelica.real
          %18 = modelica.record_create %15, %17 : !modelica.real, !modelica.real -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %19 = modelica.select (%11 : !modelica.bool), (%14 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>), (%18 : !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>) -> !modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>
          %20 = modelica.equation_side %6 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>
          %21 = modelica.equation_side %19 : tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
          modelica.equation_sides %20, %21 : tuple<!modelica.record<@PowerGridDAE.GridBase::@ComplexPU>>, tuple<!modelica.record<@PowerGridDAE.GridBase::@PowerGridDAE.Complex>>
        }
      }
    }
  }
}
