module attributes {dlti.dl_spec = #dlti.dl_spec<"dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e", "dlti.endianness" = "little", !llvm.ptr = dense<64> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f80 = dense<128> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !bmodelica.real = ["size", 64], !bmodelica.int = ["size", 64]>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  llvm.func @marco_free(!llvm.ptr)
  llvm.func @marco_malloc(i64) -> !llvm.ptr
  llvm.mlir.global internal constant @var_name_1("der_x\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @var_name_0("x\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @var_name_unknown("\00") {addr_space = 0 : i32}
  llvm.mlir.global internal constant @modelName("SimpleFirstOrder\00") {addr_space = 0 : i32}
  llvm.mlir.global private @time() {addr_space = 0 : i32} : f64 {
    %0 = llvm.mlir.undef : f64
    llvm.return %0 : f64
  }
  llvm.mlir.global private @var() {addr_space = 0 : i32} : f64 {
    %0 = llvm.mlir.undef : f64
    llvm.return %0 : f64
  }
  llvm.mlir.global private @var_0() {addr_space = 0 : i32} : f64 {
    %0 = llvm.mlir.undef : f64
    llvm.return %0 : f64
  }
  llvm.func @discreteLog(i64, i64) -> f64
  llvm.mlir.global private @timeStep() {addr_space = 0 : i32} : f64 {
    %0 = llvm.mlir.undef : f64
    llvm.return %0 : f64
  }
  llvm.func @euler_state_update_x() {
    %0 = llvm.mlir.addressof @var_0 : !llvm.ptr
    %1 = llvm.mlir.addressof @var : !llvm.ptr
    %2 = llvm.mlir.addressof @timeStep : !llvm.ptr
    %3 = llvm.load %2 : !llvm.ptr -> f64
    %4 = llvm.load %1 : !llvm.ptr -> f64
    %5 = llvm.load %0 : !llvm.ptr -> f64
    %6 = llvm.fmul %3, %5 : f64
    %7 = llvm.fadd %4, %6 : f64
    llvm.store %7, %1 : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @equation() {
    %0 = llvm.mlir.addressof @var_0 : !llvm.ptr
    %1 = llvm.mlir.constant(256 : i64) : i64
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %4 = llvm.call @externalLogReal(%2, %1) : (i64, i64) -> f64
    %5 = llvm.fsub %3, %4 : f64
    llvm.store %5, %0 : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @equation_0() {
    %0 = llvm.mlir.addressof @var : !llvm.ptr
    %1 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    llvm.store %1, %0 : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @equation_1() {
    %0 = llvm.mlir.addressof @var_0 : !llvm.ptr
    %1 = llvm.mlir.constant(256 : i64) : i64
    %2 = llvm.mlir.constant(2 : i64) : i64
    %3 = llvm.mlir.constant(1.000000e+01 : f64) : f64
    %4 = llvm.call @externalLogReal(%2, %1) : (i64, i64) -> f64
    %5 = llvm.fsub %3, %4 : f64
    llvm.store %5, %0 : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @SimpleFirstOrder_dynamic() {
    llvm.call @equation() : () -> ()
    llvm.return
  }
  llvm.func @SimpleFirstOrder_ic() {
    llvm.call @equation_0() : () -> ()
    llvm.call @equation_1() : () -> ()
    llvm.return
  }
  llvm.func @SimpleFirstOrder_schedule_state_variables() {
    llvm.call @euler_state_update_x() : () -> ()
    llvm.return
  }
  llvm.func @getModelName() -> !llvm.ptr {
    %0 = llvm.mlir.addressof @modelName : !llvm.ptr
    %1 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<17 x i8>
    llvm.return %1 : !llvm.ptr
  }
  llvm.func @getNumOfVariables() -> i64 {
    %0 = llvm.mlir.constant(2 : i64) : i64
    llvm.return %0 : i64
  }
  llvm.func @getVariableName(%arg0: i64) -> !llvm.ptr {
    %0 = llvm.mlir.addressof @var_name_1 : !llvm.ptr
    %1 = llvm.mlir.addressof @var_name_0 : !llvm.ptr
    %2 = llvm.mlir.addressof @var_name_unknown : !llvm.ptr
    %3 = llvm.getelementptr %2[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x i8>
    llvm.switch %arg0 : i64, ^bb3(%3 : !llvm.ptr) [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    %4 = llvm.getelementptr %1[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i8>
    llvm.br ^bb3(%4 : !llvm.ptr)
  ^bb2:  // pred: ^bb0
    %5 = llvm.getelementptr %0[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<6 x i8>
    llvm.br ^bb3(%5 : !llvm.ptr)
  ^bb3(%6: !llvm.ptr):  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.return %6 : !llvm.ptr
  }
  llvm.func @getVariableRank(%arg0: i64) -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    llvm.switch %arg0 : i64, ^bb3(%0 : i64) [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    llvm.br ^bb3(%0 : i64)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%0 : i64)
  ^bb3(%1: i64):  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.return %1 : i64
  }
  llvm.func @isPrintable(%arg0: i64) -> i1 {
    %0 = llvm.mlir.constant(true) : i1
    %1 = llvm.mlir.constant(false) : i1
    llvm.switch %arg0 : i64, ^bb2(%1 : i1) [
      0: ^bb1
    ]
  ^bb1:  // pred: ^bb0
    llvm.br ^bb2(%0 : i1)
  ^bb2(%2: i1):  // 2 preds: ^bb0, ^bb1
    llvm.return %2 : i1
  }
  llvm.func @getVariableNumOfPrintableRanges(%arg0: i64) -> i64 {
    %0 = llvm.mlir.constant(0 : i64) : i64
    llvm.switch %arg0 : i64, ^bb1(%0 : i64) [
    
    ]
  ^bb1(%1: i64):  // pred: ^bb0
    llvm.return %1 : i64
  }
  llvm.func @getVariablePrintableRangeBegin(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %0 = llvm.mlir.constant(-1 : i64) : i64
    llvm.switch %arg0 : i64, ^bb1(%0 : i64) [
    
    ]
  ^bb1(%1: i64):  // pred: ^bb0
    llvm.return %1 : i64
  }
  llvm.func @getVariablePrintableRangeEnd(%arg0: i64, %arg1: i64, %arg2: i64) -> i64 {
    %0 = llvm.mlir.constant(-1 : i64) : i64
    llvm.switch %arg0 : i64, ^bb1(%0 : i64) [
    
    ]
  ^bb1(%1: i64):  // pred: ^bb0
    llvm.return %1 : i64
  }
  llvm.func @getDerivative(%arg0: i64) -> i64 {
    %0 = llvm.mlir.constant(1 : i64) : i64
    %1 = llvm.mlir.constant(-1 : i64) : i64
    llvm.switch %arg0 : i64, ^bb3(%1 : i64) [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    llvm.br ^bb3(%0 : i64)
  ^bb2:  // pred: ^bb0
    llvm.br ^bb3(%1 : i64)
  ^bb3(%2: i64):  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.return %2 : i64
  }
  llvm.func @var_getter_0(%arg0: !llvm.ptr) -> f64 {
    %0 = llvm.mlir.addressof @var : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> f64
    llvm.return %1 : f64
  }
  llvm.func @var_getter_1(%arg0: !llvm.ptr) -> f64 {
    %0 = llvm.mlir.addressof @var_0 : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> f64
    llvm.return %1 : f64
  }
  llvm.func @getVariableValue(%arg0: i64, %arg1: !llvm.ptr) -> f64 {
    %0 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    llvm.switch %arg0 : i64, ^bb3(%0 : f64) [
      0: ^bb1,
      1: ^bb2
    ]
  ^bb1:  // pred: ^bb0
    %1 = llvm.call @var_getter_0(%arg1) : (!llvm.ptr) -> f64
    llvm.br ^bb3(%1 : f64)
  ^bb2:  // pred: ^bb0
    %2 = llvm.call @var_getter_1(%arg1) : (!llvm.ptr) -> f64
    llvm.br ^bb3(%2 : f64)
  ^bb3(%3: f64):  // 3 preds: ^bb0, ^bb1, ^bb2
    llvm.return %3 : f64
  }
  llvm.func @externalLogReal(%arg0: i64, %arg1: i64) -> f64 {
    %0 = llvm.mlir.zero : !llvm.ptr
    %1 = llvm.getelementptr %0[1] : (!llvm.ptr) -> !llvm.ptr, f64
    %2 = llvm.ptrtoint %1 : !llvm.ptr to i64
    %3 = llvm.call @marco_malloc(%2) : (i64) -> !llvm.ptr
    %4 = llvm.call @discreteLog(%arg0, %arg1) : (i64, i64) -> f64
    llvm.store %4, %3 : f64, !llvm.ptr
    %5 = llvm.load %3 : !llvm.ptr -> f64
    llvm.call @marco_free(%3) : (!llvm.ptr) -> ()
    llvm.return %5 : f64
  }
  llvm.func @updateNonStateVariables() {
    llvm.call @SimpleFirstOrder_dynamic() : () -> ()
    llvm.return
  }
  llvm.func @updateStateVariables(%arg0: f64) {
    %0 = llvm.mlir.addressof @timeStep : !llvm.ptr
    llvm.store %arg0, %0 : f64, !llvm.ptr
    llvm.call @SimpleFirstOrder_schedule_state_variables() : () -> ()
    llvm.return
  }
  llvm.func @solveICModel() {
    llvm.call @SimpleFirstOrder_ic() : () -> ()
    llvm.return
  }
  llvm.func @icModelBegin() {
    llvm.return
  }
  llvm.func @icModelEnd() {
    llvm.return
  }
  llvm.func @dynamicModelBegin() {
    llvm.return
  }
  llvm.func @dynamicModelEnd() {
    llvm.return
  }
  llvm.func @init() {
    %0 = llvm.mlir.addressof @var_0 : !llvm.ptr
    %1 = llvm.mlir.addressof @var : !llvm.ptr
    %2 = llvm.mlir.constant(0.000000e+00 : f64) : f64
    llvm.store %2, %1 : f64, !llvm.ptr
    llvm.store %2, %0 : f64, !llvm.ptr
    llvm.return
  }
  llvm.func @deinit() {
    llvm.return
  }
  llvm.func @getTime() -> f64 {
    %0 = llvm.mlir.addressof @time : !llvm.ptr
    %1 = llvm.load %0 : !llvm.ptr -> f64
    llvm.return %1 : f64
  }
  llvm.func @setTime(%arg0: f64) {
    %0 = llvm.mlir.addressof @time : !llvm.ptr
    llvm.store %arg0, %0 : f64, !llvm.ptr
    llvm.return
  }
}
