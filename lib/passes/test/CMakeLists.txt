set(SOURCES
		src/PassesTest.cpp)

marco_add_unittest(MARCOTest PassesTest ${SOURCES})

llvm_config(PassesTest core support)

target_link_libraries(PassesTest
		PRIVATE marco::passes)
