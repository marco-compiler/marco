set(SOURCES
		src/PassesTest.cpp)

marco_add_unittest(MARCOTest OmcToModelTest ${SOURCES})

llvm_config(OmcToModelTest core support)

target_link_libraries(OmcToModelTest
		PRIVATE marco::omcToModel)
