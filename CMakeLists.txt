cmake_minimum_required(VERSION 3.10)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.12")
    cmake_policy(SET CMP0074 NEW)
endif()

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

#if(DEFINED SIQAD_PLUGINS_ROOT)
#    set(EGS_INSTALL_DIR "${SIQAD_PLUGINS_ROOT}/exhaustive_gs")
#elseif(CMAKE_BUILD_TYPE MATCHES Release)
#    set(EGS_INSTALL_DIR "release")
#else()
#    set(EGS_INSTALL_DIR "debug")
#ENDIF()
#
#set(EGS_FILES
#    LICENSE
#    exhaustive_gs.physeng
#    src/exhaustive_gs.py
#)
#
#install(FILES ${EGS_FILES} DESTINATION ${EGS_INSTALL_DIR})
#
#if(COMMAND symlink_siqadtools AND NOT WIN32)
#    symlink_siqadtools(${EGS_INSTALL_DIR})
#endif()

# for release builds, add flag "-DCMAKE_BUILD_TYPE=Release" (without quotes) when running cmake.

project(exhaustive_gs CXX)
# TODO add way to push version number into source code

find_package(Boost COMPONENTS system filesystem thread REQUIRED)
find_package(Threads REQUIRED)

include_directories(
    .
    src
    swig
    ${Boost_INCLUDE_DIR}
)

if(SKBUILD)
    add_definitions(-Wall -g -std=c++11)
    
    find_package(SWIG REQUIRED)
    include(${SWIG_USE_FILE})

    find_package(PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_PATH})

    set_property(SOURCE swig/exhaustive_gs.i PROPERTY CPLUSPLUS ON)
    file(COPY swig/__init__.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/swig")
    file(COPY swig/wrapper_validation.py DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/swig")

    swig_add_library(exhaustive_gs LANGUAGE python SOURCES swig/exhaustive_gs.i src/exhaustive_gs.cc)

    swig_link_libraries(exhaustive_gs ${Boost_LIBRARIES} ${PYTHON_LIBRARY})

    install(TARGETS ${SWIG_MODULE_exhaustive_gs_REAL_NAME}
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../cmake-install/pyegs)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/exhaustive_gs.py
                  ${CMAKE_CURRENT_BINARY_DIR}/swig/__init__.py
                  ${CMAKE_CURRENT_BINARY_DIR}/swig/wrapper_validation.py
            DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/../cmake-install/pyegs)

else()

    if(DEFINED SIQAD_PLUGINS_ROOT)
        set(EGS_INSTALL_DIR "${SIQAD_PLUGINS_ROOT}/exhaustive_gs")
    elseif(CMAKE_BUILD_TYPE MATCHES Debug)
        set(EGS_INSTALL_DIR "debug")
    else()
        set(EGS_INSTALL_DIR "release")
    ENDIF()

    set(CMAKE_CXX_STANDARD 11)
    set(CMAKE_CXX_STANDARD_REQUIRED True)

    set(SOURCES
        src/main.cc
        src/interface.cc
        src/exhaustive_gs.cc
        src/siqadconn.cc
    )


    add_executable(exhaustive_gs ${SOURCES})

    target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})

    add_custom_command(
        TARGET exhaustive_gs POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_CURRENT_SOURCE_DIR}/exhaustive_gs.physeng
                ${CMAKE_CURRENT_BINARY_DIR})

    install(TARGETS exhaustive_gs RUNTIME DESTINATION ${EGS_INSTALL_DIR})
    install(FILES exhaustive_gs.physeng DESTINATION ${EGS_INSTALL_DIR})

endif()
