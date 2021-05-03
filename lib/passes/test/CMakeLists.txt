set(SOURCES
		src/PassesTest.cpp
		src/CleverDAETest.cpp)

marco_add_unittest(MARCOTest PassesTest ${SOURCES})

marco_link_llvm_libs(PassesTest core support)

target_link_libraries(PassesTest
		PRIVATE marco::passes)
