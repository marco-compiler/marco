include(GoogleTest)

###
### Produces the simulation test
### First argument is the base name
### Second argument is the cpp file that will produce the simulation.
### Is the test, the test must define it's own extern declaration that will
### be resolved during linking.
macro(addLowererSimulationTest simulationName sourceSrc testSrc)

	### Creates the include directory in the build tree
	file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include)

	### Copy test file
	configure_file(${sourceSrc} ${CMAKE_CURRENT_BINARY_DIR}/${simulationName}.sm COPYONLY)

	### Creates the command that will run the simulator generator and will produce the bc file
	add_custom_command(OUTPUT ${simulationName}.bc include/${simulationName}.hpp
		COMMAND modelica::simc --entry=runSimulation --publicSymbols=true --simTime=10 --dumpSim=true -bc=${simulationName}.bc -header=include/${simulationName}.hpp ${CMAKE_CURRENT_BINARY_DIR}/${simulationName}.sm
		DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${simulationName}.sm
		COMMENT "generates ${simulationName} simulation bc file"
		VERBATIM)


	### Creates the command that will run clang to generate the Obj file.
	add_custom_command(OUTPUT ${simulationName}.o
		COMMAND clang-9 -o ${simulationName}.o -fPIC -c ${simulationName}.bc
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

addLowererSimulationTest(typeSimulation ${CMAKE_CURRENT_SOURCE_DIR}/src/TypeTest.sm src/TypeSimulationTest.cpp)
addLowererSimulationTest(arraySimulation ${CMAKE_CURRENT_SOURCE_DIR}/src/ArrayTest.sm src/ArraySimulationTest.cpp)
addLowererSimulationTest(floatSimulation ${CMAKE_CURRENT_SOURCE_DIR}/src/FloatTest.sm src/FloatSimulationTest.cpp)
addLowererSimulationTest(condSimulation ${CMAKE_CURRENT_SOURCE_DIR}/src/CondTest.sm src/CondSimulationTest.cpp)
addLowererSimulationTest(callSimulation ${CMAKE_CURRENT_SOURCE_DIR}/src/CallTest.sm src/CallSimulationTest.cpp)

