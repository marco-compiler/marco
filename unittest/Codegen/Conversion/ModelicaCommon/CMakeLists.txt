set(SOURCES
    TypeConverterTest.cpp)

marco_add_unittest(ModelicaCommonConversionTest ${SOURCES})

marco_link_llvm_libs(ModelicaCommonConversionTest Core Support)

target_link_libraries(ModelicaCommonConversionTest
        PRIVATE MLIRModelicaCommonConversion
        PRIVATE marco::codegen)
