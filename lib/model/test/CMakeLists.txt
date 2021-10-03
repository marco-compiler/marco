set(SOURCES
		src/ModelTest.cpp
		src/ModLexerStateMachineTest.cpp
		src/ModParserTest.cpp
		src/ModMatchersTest.cpp
		src/VectorAccessTest.cpp
		src/ModEquationTest.cpp
		src/ModConstTest.cpp)

marco_add_unittest(MARCOTest ModelTest ${SOURCES})

llvm_config(ModelTest core support)

target_link_libraries(ModelTest
		PRIVATE marco::model)
