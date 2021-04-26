set(SOURCES
		src/ModelTest.cpp
		src/ModLexerStateMachineTest.cpp
		src/ModParserTest.cpp
		src/ModMatchersTest.cpp
		src/VectorAccessTest.cpp
		src/ModEquationTest.cpp
		src/ModConstTest.cpp
		src/SymbolicDifferentiationTest.cpp)

marco_add_unittest(MARCOTest ModelTest ${SOURCES})

marco_link_llvm_libs(ModelTest core support)

target_link_libraries(ModelTest
		PRIVATE marco::model)
