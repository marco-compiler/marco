include(GoogleTest)
add_subdirectory(testingSimulationGenerator)

add_executable(lowererTest src/LowererTest.cpp)
add_executable(modelica::lowererTest ALIAS lowererTest) 
target_link_libraries(lowererTest PRIVATE gtest gtest_main modelica::lowerer)
target_include_directories(lowererTest PUBLIC include PRIVATE src)
target_compile_features(lowererTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      lowererTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)


add_executable(intTestGenerator src/IntTestGenerator.cpp)
add_executable(modelica::intTestGenerator ALIAS intTestGenerator)
target_link_libraries(intTestGenerator PRIVATE modelica::testingSimulationGenerator)
target_compile_features(intTestGenerator PUBLIC cxx_std_17)

add_custom_command(OUTPUT intSimulation.bc
					COMMAND modelica::intTestGenerator -bc=intSimulation.bc
					COMMENT "generates int simulation bc file"
					DEPENDS modelica::intTestGenerator 
					VERBATIM)


add_custom_command(OUTPUT intSimulation.o
	COMMAND clang -o intSimulation.o -fPIC -c intSimulation.bc
					COMMENT "generates int simulation object file"
					DEPENDS intSimulation.bc
					VERBATIM)

add_custom_target(genIntSim ALL DEPENDS intSimulation.o)

add_executable(intSimulationTest src/IntSimulationTest.cpp)
add_executable(modelica::intSimulationTest ALIAS intSimulationTest) 
target_link_libraries(intSimulationTest PRIVATE gtest gtest_main ${CMAKE_CURRENT_BINARY_DIR}/intSimulation.o modelica::runtime)
target_include_directories(intSimulationTest PUBLIC include PRIVATE src)
target_compile_features(intSimulationTest PUBLIC cxx_std_17)
add_dependencies(intSimulationTest genIntSim)

gtest_add_tests(TARGET      intSimulationTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

