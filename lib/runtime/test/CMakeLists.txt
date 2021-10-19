set(SOURCES
		src/ArrayDescriptorTest.cpp
		src/BuiltInFunctionsTest.cpp
        src/UtilityFunctionsTest.cpp)

marco_add_unittest(MARCOTest RuntimeTest ${SOURCES})

marco_link_llvm_libs(RuntimeTest core support)

target_link_libraries(RuntimeTest
		PRIVATE marco::runtime)
