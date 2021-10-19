set(SOURCES
		src/PassesTest.cpp)

marco_add_unittest(MARCOTest OmcToModelTest ${SOURCES})

marco_link_llvm_libs(OmcToModelTest core support)

target_link_libraries(OmcToModelTest
		PRIVATE marco::omcToModel)
