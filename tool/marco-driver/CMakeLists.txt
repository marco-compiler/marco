set(SOURCES
    marco.cpp
    marco-mc1.cpp)

set(LLVM_LIBS
    LLVMWindowsManifest
    LLVMXRay
    LLVMLibDriver
    LLVMDlltoolDriver
    LLVMCoverage
    LLVMLineEditor
    LLVMXCoreDisassembler
    LLVMXCoreCodeGen
    LLVMXCoreDesc
    LLVMXCoreInfo
    LLVMX86Disassembler
    LLVMX86AsmParser
    LLVMX86CodeGen
    LLVMX86Desc
    LLVMX86Info
    LLVMWebAssemblyDisassembler
    LLVMWebAssemblyAsmParser
    LLVMWebAssemblyCodeGen
    LLVMWebAssemblyDesc
    LLVMWebAssemblyUtils
    LLVMWebAssemblyInfo
    LLVMSystemZDisassembler
    LLVMSystemZAsmParser
    LLVMSystemZCodeGen
    LLVMSystemZDesc
    LLVMSystemZInfo
    LLVMSparcDisassembler
    LLVMSparcAsmParser
    LLVMSparcCodeGen
    LLVMSparcDesc
    LLVMSparcInfo
    LLVMRISCVDisassembler
    LLVMRISCVAsmParser
    LLVMRISCVCodeGen
    LLVMRISCVDesc
    LLVMRISCVInfo
    LLVMPowerPCDisassembler
    LLVMPowerPCAsmParser
    LLVMPowerPCCodeGen
    LLVMPowerPCDesc
    LLVMPowerPCInfo
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
    LLVMMSP430Disassembler
    LLVMMSP430AsmParser
    LLVMMSP430CodeGen
    LLVMMSP430Desc
    LLVMMSP430Info
    LLVMMipsDisassembler
    LLVMMipsAsmParser
    LLVMMipsCodeGen
    LLVMMipsDesc
    LLVMMipsInfo
    LLVMLanaiDisassembler
    LLVMLanaiCodeGen
    LLVMLanaiAsmParser
    LLVMLanaiDesc
    LLVMLanaiInfo
    LLVMHexagonDisassembler
    LLVMHexagonCodeGen
    LLVMHexagonAsmParser
    LLVMHexagonDesc
    LLVMHexagonInfo
    LLVMBPFDisassembler
    LLVMBPFAsmParser
    LLVMBPFCodeGen
    LLVMBPFDesc
    LLVMBPFInfo
    LLVMAVRDisassembler
    LLVMAVRAsmParser
    LLVMAVRCodeGen
    LLVMAVRDesc
    LLVMAVRInfo
    LLVMARMDisassembler
    LLVMARMAsmParser
    LLVMARMCodeGen
    LLVMARMDesc
    LLVMARMUtils
    LLVMARMInfo
    LLVMAMDGPUDisassembler
    LLVMAMDGPUAsmParser
    LLVMAMDGPUCodeGen
    LLVMAMDGPUDesc
    LLVMAMDGPUUtils
    LLVMAMDGPUInfo
    LLVMAArch64Disassembler
    LLVMAArch64AsmParser
    LLVMAArch64CodeGen
    LLVMAArch64Desc
    LLVMAArch64Utils
    LLVMAArch64Info
    LLVMOrcJIT
    LLVMMCJIT
    LLVMJITLink
    LLVMInterpreter
    LLVMExecutionEngine
    LLVMRuntimeDyld
    LLVMOrcTargetProcess
    LLVMOrcShared
    LLVMDWP
    LLVMSymbolize
    LLVMDebugInfoPDB
    LLVMDebugInfoGSYM
    LLVMOption
    LLVMObjectYAML
    LLVMMCA
    LLVMMCDisassembler
    LLVMLTO
    LLVMPasses
    LLVMCFGuard
    LLVMCoroutines
    LLVMObjCARCOpts
    LLVMipo
    LLVMVectorize
    LLVMLinker
    LLVMInstrumentation
    LLVMFrontendOpenMP
    LLVMFrontendOpenACC
    LLVMExtensions
    LLVMDWARFLinker
    LLVMGlobalISel
    LLVMMIRParser
    LLVMAsmPrinter
    LLVMDebugInfoMSF
    LLVMDebugInfoDWARF
    LLVMSelectionDAG
    LLVMCodeGen
    LLVMIRReader
    LLVMAsmParser
    LLVMInterfaceStub
    LLVMFileCheck
    LLVMFuzzMutate
    LLVMTarget
    LLVMScalarOpts
    LLVMInstCombine
    LLVMAggressiveInstCombine
    LLVMTransformUtils
    LLVMBitWriter
    LLVMAnalysis
    LLVMProfileData
    LLVMObject
    LLVMTextAPI
    LLVMMCParser
    LLVMMC
    LLVMDebugInfoCodeView
    LLVMBitReader
    LLVMCore
    LLVMRemarks
    LLVMBitstreamReader
    LLVMBinaryFormat
    LLVMTableGen
    LLVMSupport
    LLVMDemangle)

marco_add_tool(marco-driver ${SOURCES})

target_link_libraries(marco-driver
    PRIVATE marco::frontend
    PRIVATE marco::frontendTool)

if(MSVC)
clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver
    ${LLVM_LIBS}) # needed to avoid undefined references
else()
clang_target_link_libraries(marco-driver
    PRIVATE
    clangDriver)
endif()

install(TARGETS marco-driver RUNTIME DESTINATION bin)

# This shell script will only work in a POSIX shell.
if (NOT WIN32)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/marco
      DESTINATION ${MARCO_TOOLS_DIR}
      FILE_PERMISSIONS OWNER_EXECUTE OWNER_READ OWNER_WRITE)

  install(PROGRAMS ${CMAKE_CURRENT_SOURCE_DIR}/marco DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
