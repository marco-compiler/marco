set(SOURCES
		TestingUtils.cpp.backup
    matching/LinSolverTest.cpp.backup
		matching/MatchingTest.cpp.backup
		matching/SCCCollapsingTest.cpp.backup
		matching/ScheduleTest.cpp.backup
		matching/VVarDependencyGraphTest.cpp.backup
		model/EquationTest.cpp.backup
		model/ReferenceMatcherTest.cpp.backup
		model/VectorAccessTest.cpp.backup
		passes/SolveModelTest..backup)

marco_add_unittest(MARCOTest GtestTest ${SOURCES})

llvm_config(GtestTest core support)

target_link_libraries(GtestTest
		PRIVATE marco::mlirlowerer)
