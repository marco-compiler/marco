set(SOURCES
		src/ArrayDescriptorTest.cpp
		src/BuiltInFunctionsTest.cpp
        src/UtilityFunctionsTest.cpp)

marco_add_unittest(MARCOTest RuntimeTest ${SOURCES})

llvm_config(RuntimeTest core support)

target_link_libraries(RuntimeTest
		PRIVATE marco::runtime)
