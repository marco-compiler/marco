module {
memref.global "private" constant @__constant_10x20xf64 : memref<10x20xf64> = dense<0.000000e+00> {alignment = 64 : i64}
  memref.global "private" @time : memref<f64> = uninitialized
  memref.global "private" @var : memref<10x20xf64> = uninitialized
  memref.global "private" @timeStep : memref<f64> = uninitialized
  func.func @equation(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
    %cst = arith.constant 1.000000e+00 : f64
    %c-1 = arith.constant -1 : index
    %0 = memref.get_global @var : memref<10x20xf64>
    affine.for %arg4 = %arg0 to %arg1 {
      affine.for %arg5 = %arg2 to %arg3 {
        %1 = arith.addi %arg4, %c-1 : index
        %2 = arith.addi %arg5, %c-1 : index
        %subview = memref.subview %0[%1, %2] [1, 1] [1, 1] : memref<10x20xf64> to memref<f64, strided<[], offset: ?>>
        memref.store %cst, %subview[] : memref<f64, strided<[], offset: ?>>
      }
    }
    return
  }
  func.func @equation_0(%arg0: index, %arg1: index, %arg2: index, %arg3: index) {
    %cst = arith.constant 1.000000e+00 : f64
    %c-1 = arith.constant -1 : index
    %0 = memref.get_global @var : memref<10x20xf64>
    affine.for %arg4 = %arg0 to %arg1 {
      affine.for %arg5 = %arg2 to %arg3 {
        %1 = arith.addi %arg4, %c-1 : index
        %2 = arith.addi %arg5, %c-1 : index
        %subview = memref.subview %0[%1, %2] [1, 1] [1, 1] : memref<10x20xf64> to memref<f64, strided<[], offset: ?>>
        memref.store %cst, %subview[] : memref<f64, strided<[], offset: ?>>
      }
    }
    return
  }
}
