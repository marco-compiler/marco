set(SOURCES
		TestingUtils.cpp
		matching/LinSolverTest.cpp
		matching/MatchingTest.cpp
		matching/SCCCollapsingTest.cpp
		matching/ScheduleTest.cpp
		matching/VVarDependencyGraphTest.cpp
		model/EquationTest.cpp
		model/ReferenceMatcherTest.cpp
		model/VectorAccessTest.cpp
		passes/SolveModelTest.cpp)

marco_add_unittest(MARCOTest GtestTest ${SOURCES})

llvm_config(GtestTest core support)

target_link_libraries(GtestTest
		PRIVATE marco::ida
		PRIVATE marco::mlirlowerer
		PRIVATE sundials)
