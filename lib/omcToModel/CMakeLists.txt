set(SOURCES
		src/OmcToModelPass.cpp)

marco_add_library(omcToModel SHARED ${SOURCES})

llvm_config(omcToModel USE_SHARED core support)

target_link_libraries(omcToModel
		PUBLIC marco::frontend
		PUBLIC marco::model)

add_subdirectory(test)
