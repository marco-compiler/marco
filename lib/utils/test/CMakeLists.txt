set(SOURCES
		src/DefaultTest.cpp
		src/LexerTest.cpp
		src/NumbersLexerTest.cpp
		src/IntervalTest.cpp
		src/IRangeTest.cpp
		src/ScopeGuard.cpp
		src/VariableFilterTest.cpp)

marco_add_unittest(MARCOTest UtilsTest ${SOURCES})

marco_link_llvm_libs(UtilsTest core support)

target_link_libraries(UtilsTest
		PRIVATE marco::utils)
