.code

PUBLIC msvc_asm_sqrt

msvc_asm_sqrt PROC
  movsd QWORD PTR 8[rsp], xmm0
  fld QWORD PTR 8[rsp]
  fsqrt
  fstp QWORD PTR [rdx]
  ret 0
msvc_asm_sqrt ENDP

END