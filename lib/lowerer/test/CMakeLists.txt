include(GoogleTest)

###
### Produces the simulation test
### First argument is the base name
### Second argument is the cpp file that will produce the simulation.
### Is the test, the test must define it's own extern declaration that will
### be resolved during linking.
macro(addLowererSimulationTest simulationName generatorSrc testSrc)

	### Creates the include directory in the build tree
	file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

	### Creates the executable that will generate the simulation
	### Object file.
	add_executable(${simulationName}Generator ${generatorSrc})
	add_executable(modelica::${simulationName}Generator ALIAS ${simulationName}Generator)
	target_link_libraries(${simulationName}Generator PRIVATE modelica::simulationGenerator)
	target_compile_features(${simulationName}Generator PUBLIC cxx_std_17)

	### Creates the command that will run the simulator generator and will produce the bc file
	add_custom_command(OUTPUT ${simulationName}.bc include/${simulationName}.hpp
						COMMAND modelica::${simulationName}Generator -bc=${simulationName}.bc -header=include/${simulationName}.hpp
						COMMENT "generates ${simulationName} simulation bc file"
						DEPENDS modelica::${simulationName}Generator
						VERBATIM)


	### Creates the command that will run clang to generate the Obj file.
	add_custom_command(OUTPUT ${simulationName}.o
		COMMAND clang -o ${simulationName}.o -fPIC -c ${simulationName}.bc
						COMMENT "generates ${simulationName} simulation object file"
						DEPENDS ${simulationName}.bc
						VERBATIM)

	### Creates a target so that other targets can depend on this one to make sure
	### the obj file is up to date
	add_custom_target(gen${simulationName} ALL DEPENDS ${simulationName}.o)

	### Creates a target so that other targets can depend on this one to make sure
	### the header file is up to date
	add_custom_target(gen${simulationName}Header ALL DEPENDS include/${simulationName}.hpp)

	### Creates the actual test 
	add_executable(${simulationName}Test ${testSrc})
	add_executable(modelica::${simulationName}Test ALIAS ${simulationName}Test) 
	target_link_libraries(${simulationName}Test PRIVATE gtest gtest_main ${CMAKE_CURRENT_BINARY_DIR}/${simulationName}.o modelica::runtime)
	target_include_directories(${simulationName}Test PUBLIC include PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/include)
	target_compile_features(${simulationName}Test PUBLIC cxx_std_17)
	add_dependencies(${simulationName}Test gen${simulationName})
	add_dependencies(${simulationName}Test gen${simulationName}Header)

	gtest_add_tests(TARGET		${simulationName}Test
					TEST_SUFFIX .noArgs
					TEST_LIST   noArgsTests
	)
endmacro(addLowererSimulationTest)

addLowererSimulationTest(typeSimulation src/TypeTestGenerator.cpp src/TypeSimulationTest.cpp)
addLowererSimulationTest(arraySimulation src/ArrayTestGenerator.cpp src/ArraySimulationTest.cpp)
addLowererSimulationTest(floatSimulation src/FloatTestGenerator.cpp src/FloatSimulationTest.cpp)
addLowererSimulationTest(condSimulation src/CondTestGenerator.cpp src/CondSimulationTest.cpp)
addLowererSimulationTest(callSimulation src/CallTestGenerator.cpp src/CallSimulationTest.cpp)

