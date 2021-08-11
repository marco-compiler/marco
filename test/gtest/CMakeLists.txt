modelicaAddTest(
	mlirlowerer
	TestingUtils.cpp
	ida/IdaSolverTest.cpp
	matching/LinSolverTest.cpp
	matching/MatchingTest.cpp
	matching/SCCCollapsingTest.cpp
	matching/ScheduleTest.cpp
	matching/VVarDependencyGraphTest.cpp
	model/EquationTest.cpp
	model/ReferenceMatcherTest.cpp
	model/VectorAccessTest.cpp
	passes/SolveModelTest.cpp
)

target_link_libraries(mlirlowererTest PRIVATE sundials)
