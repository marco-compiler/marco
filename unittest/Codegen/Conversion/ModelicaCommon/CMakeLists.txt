set(SOURCES
    TypeConverterTest.cpp)

set(LLVM_LINK_COMPONENTS
    Core
    Support)

marco_add_unittest(ModelicaCommonConversionTest ${SOURCES})

target_link_libraries(ModelicaCommonConversionTest
        PRIVATE MLIRBaseModelicaCommonConversion
        PRIVATE marco::codegen)
