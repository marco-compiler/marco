.code

PUBLIC msvc_asm_sin

angle_on_stack$ = 8

msvc_asm_sin PROC

  movsd QWORD PTR angle_on_stack$[rsp], xmm0
  fld QWORD PTR angle_on_stack$[rsp]
  fsincos
  fstp QWORD PTR [r8]
  fstp QWORD PTR [rdx]
  ret 0

msvc_asm_sin ENDP

END