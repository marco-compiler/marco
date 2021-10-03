set(SOURCES
		src/ArrayDescriptorTest.cpp
		src/BuiltInFunctionsTest.cpp
		src/RuntimeTest.cpp)

marco_add_unittest(MARCOTest RuntimeTest ${SOURCES})

llvm_config(RuntimeTest core support)

target_link_libraries(RuntimeTest
		PRIVATE marco::runtime)
