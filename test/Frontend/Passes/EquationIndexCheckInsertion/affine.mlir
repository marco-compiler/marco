// RUN: modelica-opt %s --split-input-file --equation-index-check-insertion | FileCheck %s

// CHECK: #[[condition:.*]] = affine_set<(d0)[s0, s1] : (d0 - s0 >= 0, -d0 + s1 + 1 >= 0)>
// CHECK-LABEL: @singleLoop
// CHECK-SAME:  (%[[lb:.*]]: index, %[[ub:.*]]: index)
// CHECK:       affine.for %[[i:.*]] = %[[lb]] to %[[ub]]
// CHECK-NEXT:      affine.if #[[condition]](%[[i]])[%[[lb]], %[[ub]]]
// CHECK-NEXT:          func.call @foo(%[[i]])

func.func @singleLoop(%lb: index, %ub: index) attributes {equation_function} {
    affine.for %i = %lb to %ub {
        func.call @foo(%i) : (index) -> ()
    }

    return
}

func.func @foo(%arg0: index) {
  return
}

// -----

// CHECK: #[[condition:.*]] = affine_set<(d0, d1, d2)[s0, s1, s2, s3, s4, s5] : (d0 - s0 >= 0, -d0 + s1 + 1 >= 0, d1 - s2 >= 0, -d1 + s3 + 1 >= 0, d2 - s4 >= 0, -d2 + s5 + 1 >= 0)>
// CHECK-LABEL: @nestedLoops
// CHECK-SAME:  (%[[lb1:.*]]: index, %[[ub1:.*]]: index, %[[lb2:.*]]: index, %[[ub2:.*]]: index, %[[lb3:.*]]: index, %[[ub3:.*]]: index)
// CHECK:       affine.for %[[i:.*]] = %[[lb1]] to %[[ub1]]
// CHECK-NEXT:      affine.for %[[j:.*]] = %[[lb2]] to %[[ub2]]
// CHECK-NEXT:          affine.for %[[k:.*]] = %[[lb3]] to %[[ub3]]
// CHECK-NEXT:              affine.if #[[condition]](%[[i]], %[[j]], %[[k]])[%[[lb1]], %[[ub1]], %[[lb2]], %[[ub2]], %[[lb3]], %[[ub3]]]
// CHECK-NEXT:                  func.call @foo(%[[i]], %[[j]], %[[k]])

func.func @nestedLoops(%lb1: index, %ub1: index, %lb2: index, %ub2: index, %lb3: index, %ub3: index) attributes {equation_function} {
    affine.for %i = %lb1 to %ub1 {
        affine.for %j = %lb2 to %ub2 {
            affine.for %k = %lb3 to %ub3 {
                func.call @foo(%i, %j, %k) : (index, index, index) -> ()
            }
        }
    }

    return
}

func.func @foo(%arg0: index, %arg1: index, %arg2: index) {
  return
}
