// RUN: marco --omc-bypass --model=SimpleFirstOrder --end-time=1 -o %basename_t %s
// RUN: ./%basename_t --precision=6 | FileCheck %s

// CHECK: time;x
// CHECK-NEXT: 0.000000;0.000000
// CHECK-NEXT: 0.100000;0.100000
// CHECK-NEXT: 0.200000;0.190000
// CHECK-NEXT: 0.300000;0.271000
// CHECK-NEXT: 0.400000;0.343900
// CHECK-NEXT: 0.500000;0.409510
// CHECK-NEXT: 0.600000;0.468559
// CHECK-NEXT: 0.700000;0.521703
// CHECK-NEXT: 0.800000;0.569533
// CHECK-NEXT: 0.900000;0.612580
// CHECK-NEXT: 1.000000;0.651322

model SimpleFirstOrder
    Real x;
equation
    der(x) = 1 - x;
end SimpleFirstOrder;
