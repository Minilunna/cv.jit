
##  Projeto LEI
##  Claudia Pires - 1303334
##  2022
##  CMakeList.txt

include(${CMAKE_CURRENT_SOURCE_DIR}/../cvjit-pretarget.cmake)

unset(PROJECT_SOURCES)
if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/max.${PROJECT_NAME}.cpp)
    set(PROJECT_SOURCES ${PROJECT_SOURCES} max.${PROJECT_NAME}.cpp)
endif()

if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpp)
    set(PROJECT_SOURCES ${PROJECT_SOURCES} ${PROJECT_NAME}.cpp)
endif()

## OpenCV libraries
if (WIN32)
	file(GLOB_RECURSE ZXING_LIBS "${ZXING_INSTALL_DIR}/lib/*.lib")
elseif(APPLE)
	file(GLOB ZXING_LIBS "${ZXING_INSTALL_DIR}/lib/*.a")
endif()

include_directories("${ZXING_INSTALL_DIR}/include")

set(PROJECT_SOURCES ${PROJECT_SOURCES}  ${CMAKE_CURRENT_SOURCE_DIR}/zxing/MatSource.cpp)

set (EXTRA_LIBS ${EXTRA_LIBS} ${ZXING_LIBS})

add_library( 
	${PROJECT_NAME} 
	MODULE
	${PROJECT_SOURCES}
)

include(${CMAKE_CURRENT_SOURCE_DIR}/../cvjit-posttarget.cmake)